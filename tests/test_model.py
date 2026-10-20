import numpy as np
import pandas as pd
import pytest
import torch
import pytorch_lightning as pl
from diffcl.training import VARIANTS,ReleaseLitModel,DataBundle,ValidationHistory,load_trainable,choose_duration
from dataset import _component_mutation_masks
from run_experiment import prediction_frame
from lit_model import MODEL_BATCH_KEYS,MUTATION_MASK_KEYS

@pytest.mark.parametrize('variant',VARIANTS)
def test_variant_gradients_and_frozen_backbone(toy,variant):
    cfg,df,h=toy;cfg={**cfg,'variant':variant};bundle=DataBundle(cfg,df,h)
    net=ReleaseLitModel(cfg).train();batch=next(iter(bundle.loader(df,'train')))
    aux=net._forward_from_batch(batch,return_aux=True)
    loss=net._objective(aux['pred'],batch['labels'],aux,batch)['total'];loss.backward()
    assert torch.isfinite(loss) and aux['pred'].shape==(4,)
    assert all(p.grad is None for p in net.model.encoder.parameters())
    assert not net.model.encoder.training
    assert net.model.prediction_head[-1].weight.grad.abs().sum()>0
    if variant!='no_cl': assert float(aux['cl1_loss'])>0 and float(aux['cl2_loss'])>0
    if variant=='no_cl1': assert net.cl1_weight==0 and net.cl2_weight>0
    if variant=='no_cl2': assert net.cl2_weight==0 and net.cl1_weight>0
    if variant=='no_cl': assert not net.use_contrastive
    if variant=='simple_diff': assert net.model.state_difference_fusion.net[1].in_features==48

@pytest.mark.parametrize('variant',VARIANTS)
def test_lightning_train_validate_and_export(toy,variant,tmp_path):
    cfg,df,h=toy;cfg={**cfg,'variant':variant};bundle=DataBundle(cfg,df,h)
    history=ValidationHistory();net=ReleaseLitModel(cfg)
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,logger=False,callbacks=[history],
        enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
    trainer.fit(net,bundle.loader(df,'train',True),bundle.loader(df,'val'))
    assert len(history.rows)==1 and np.isfinite(history.rows[0]['rmse'])
    chosen=choose_duration(history.rows,dict(selection_start_epoch=1,min_val_pred_std_ratio=0,val_rmse_safe_fraction=.03,min_refit_epochs=1,max_refit_epochs=1))
    assert chosen['refit_epochs']==1
    trainer.test(net,bundle.loader(df,'test'),verbose=False)
    out=prediction_frame(net.get_last_test_predictions(),1)
    np.testing.assert_allclose(out.y_true,df.ddG,rtol=0,atol=1e-6)
    assert list(out.source_index)==list(range(4)) and np.isfinite(out.y_pred).all()

def test_checkpoint_roundtrip(toy):
    cfg,df,h=toy;net=ReleaseLitModel(cfg).eval();bundle=DataBundle(cfg,df,h);batch=next(iter(bundle.loader(df,'test')))
    state={n:p.detach().clone() for n,p in net.named_parameters() if p.requires_grad}
    other=ReleaseLitModel(cfg).eval();load_trainable(other,state)
    torch.testing.assert_close(net._forward_from_batch(batch,return_aux=False),other._forward_from_batch(batch,return_aux=False),rtol=0,atol=0)
    with pytest.raises(ValueError): load_trainable(other,{**state,'unrecognized':torch.zeros(1)})

def test_local_window_stops_at_chain_boundary():
    mut,window,n=_component_mutation_masks(('AAAA','CCCC'),('AAAV','CCCC'),8)
    assert n==1 and sum(mut)==1 and list(window)==[True]*4+[False]*4

def test_empty_negative_set_is_differentiable(toy):
    cfg,df,h=toy;net=ReleaseLitModel(cfg);bundle=DataBundle(cfg,df.iloc[:2],h)
    batch=next(iter(bundle.loader(df.iloc[:2],'train')));aux=net._forward_from_batch(batch,return_aux=True)
    assert float(aux['cl1_loss'])==0 and float(aux['cl2_loss'])==0
    net._objective(aux['pred'],batch['labels'],aux,batch)['total'].backward()

def test_long_sequence_is_not_truncated(toy):
    cfg,df,h=toy;cfg={**cfg,'use_contrastive':False}
    frame=df.iloc[:1].copy();frame['a']='A'*90;frame['a_mut']='A'*89+'V'
    bundle=DataBundle(cfg,None,target_statistics=(0,1));batch=next(iter(bundle.loader(frame,'test')))
    assert batch['wt_ab_inputs_ids'].shape[1]>=92
    net=ReleaseLitModel(cfg).eval();out=net._forward_from_batch(batch,return_aux=False)
    assert torch.isfinite(out).all()

def test_formal_backend_cannot_silently_fallback(toy,monkeypatch):
    import model
    monkeypatch.setattr(model,'_HAS_MAMBA',False)
    with pytest.raises(ImportError): ReleaseLitModel({**toy[0],'allow_mamba_fallback':False,'strict_reviewer_mode':True})

def test_inference_checkpoint_statistics_and_row_comparison(toy,tmp_path):
    from argparse import Namespace
    from predict import predict
    cfg,df,h=toy;cfg={**cfg,'variant':'cnn','use_contrastive':False}
    net=ReleaseLitModel(cfg).eval();bundle=DataBundle(cfg,df,h)
    checkpoint=tmp_path/'model.pt'
    torch.save({'config':cfg,'state_dict':{n:p.detach() for n,p in net.named_parameters() if p.requires_grad},
                'target_statistics':{'mean':bundle.mean,'std':bundle.std}},checkpoint)
    df.to_csv(tmp_path/'input.csv',index=False)
    args=Namespace(device='cpu',checkpoint=checkpoint,backbone=cfg['model_locate'],statistics=None,
                   input=tmp_path/'input.csv',output=tmp_path/'predictions.csv',batch_size=4,compare=None,atol=1e-4)
    output=predict(args)
    assert len(output)==len(df)
    args.compare=args.output;args.output=tmp_path/'repeat.csv'
    repeat=predict(args)
    np.testing.assert_allclose(output.y_pred,repeat.y_pred,rtol=0,atol=0)
