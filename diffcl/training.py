"""Release training interfaces; the deposited full-model implementation is unchanged."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
import pytorch_lightning as pl
from dataset import ProteinMutationDataset, MutationAwareCollator, MutationAwareBatchSampler, DirectHomologyIndex, infer_schema
from lit_model import LitModel
from model import masked_mean
from diffcl.common import validate_frame

VARIANTS=('full','cnn','no_cl1','no_cl2','no_cl','simple_diff')

class SimpleDifferenceFusion(nn.Module):
    def __init__(self,d,dropout):
        super().__init__();self.d=d
        self.net=nn.Sequential(nn.LayerNorm(3*d),nn.Linear(3*d,2*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d,d),nn.LayerNorm(d))
    def forward(self,x):
        # Incoming blocks: WT, MT, signed delta, absolute delta, product, local.
        return self.net(torch.cat((x[...,:2*self.d],x[...,5*self.d:]),dim=-1))

class CNNBranch(nn.Module):
    """Declared residual CNN comparator: one shared kernel-3 sequence block."""
    def __init__(self,d,dropout):
        super().__init__();self.norm=nn.LayerNorm(d);self.conv=nn.Conv1d(d,d,3,padding=1)
        self.linear=nn.Linear(d,d);self.drop=nn.Dropout(dropout);self.out_norm=nn.LayerNorm(d)
    def forward(self,x,mask):
        m=mask.unsqueeze(-1).to(x.dtype)
        h=self.conv((self.norm(x)*m).transpose(1,2)).transpose(1,2)
        x=self.out_norm(x+self.drop(self.linear(torch.nn.functional.gelu(h))))*m
        return x,masked_mean(x,mask.bool())

class ReleaseLitModel(LitModel):
    def __init__(self,cfg):
        cfg=dict(cfg);variant=cfg.get('variant','full')
        if variant not in VARIANTS: raise ValueError(variant)
        if variant in ('no_cl1','no_cl'): cfg['cl1_weight']=0.
        if variant in ('no_cl2','no_cl'): cfg['cl2_weight']=0.
        if variant=='no_cl': cfg['use_contrastive']=False
        # Construct the same fusion path without requiring CUDA Mamba for CNN.
        strict=cfg.get('strict_reviewer_mode',True)
        if variant=='cnn': cfg.update(allow_mamba_fallback=True,strict_reviewer_mode=False)
        super().__init__(cfg);self.variant=variant
        if variant=='cnn': self.model.shared_state_encoder.mamba=CNNBranch(self.model.model_dim,self.model.dropout)
        self.strict_reviewer_mode=strict;self.cfg['strict_reviewer_mode']=strict
        if variant=='simple_diff': self.model.state_difference_fusion=SimpleDifferenceFusion(self.model.model_dim,self.model.dropout)
    def on_fit_start(self):
        if self.variant!='cnn': return super().on_fit_start()
        if any(p.requires_grad for p in self.model.encoder.parameters()): raise RuntimeError('ESM must remain frozen')
        if self.model.encoder.training: raise RuntimeError('Frozen ESM must be in evaluation mode')
        if any(isinstance(x,nn.modules.batchnorm._BatchNorm) for x in self.modules()): raise RuntimeError('Unexpected BatchNorm')
        if not isinstance(self.model.shared_state_encoder.mamba,CNNBranch): raise RuntimeError('Invalid CNN branch')

class DataBundle:
    def __init__(self,cfg,train,homology=None,target_statistics=None):
        self.cfg=cfg
        if target_statistics is None:
            validate_frame(train,'training')
            self.mean=float(train.ddG.mean());self.std=float(train.ddG.std(ddof=1))
            if not np.isfinite(self.std) or self.std<=0: raise ValueError('Training labels need nonzero sample variance')
        else: self.mean,self.std=map(float,target_statistics)
        self.tokenizer=AutoTokenizer.from_pretrained(cfg['model_locate'],local_files_only=True)
        self.max_length=int(AutoConfig.from_pretrained(cfg['model_locate'],local_files_only=True).max_position_embeddings)
        if homology is None:
            if cfg.get('use_contrastive',False): raise ValueError('Training with CL requires a complete homology index')
            self.homology=ExactOnly()
        else:
            h=Path(homology)
            self.homology=DirectHomologyIndex(h/'sequences.fasta',h/'all_vs_all.tsv',.90,.80)
        self.collator=MutationAwareCollator(self.tokenizer,self.max_length,self.homology,self.mean,self.std,True,False,'sliding_window')
    def dataset(self,frame,split):
        frame=frame.copy()
        frame['_ab645_row_id']=frame.record_id if 'record_id' in frame else np.arange(len(frame))
        return ProteinMutationDataset(frame,infer_schema(frame),split,self.mean,self.std,True,self.cfg.get('mutation_window_radius',8),True)
    def loader(self,frame,split,training=False):
        ds=self.dataset(frame,split)
        if hasattr(self.homology,'validate_coverage'):
            self.homology.validate_coverage(ds.all_component_sequences(),strict=True)
        common=dict(num_workers=self.cfg.get('num_workers',0),pin_memory=torch.cuda.is_available(),collate_fn=self.collator)
        if training:
            sampler=MutationAwareBatchSampler(ds,self.homology,self.cfg['batch_size'],self.cfg['seed'],
                self.cfg.get('complex_temper_alpha',.5),self.cfg.get('same_complex_pairs_per_batch',2),True)
            return DataLoader(ds,batch_sampler=sampler,**common)
        return DataLoader(ds,batch_size=self.cfg.get('eval_batch_size',self.cfg['batch_size']),shuffle=False,**common)

class ExactOnly:
    """Inference does not construct or optimize a contrastive objective."""
    def are_related(self,a,b): return bool(a and a==b)
    def components_related(self,a,b): return any(self.are_related(s,t) for s in a for t in b)

class ValidationHistory(pl.Callback):
    def __init__(self): self.rows=[]
    def on_validation_end(self,trainer,module):
        if not trainer.sanity_checking:
            self.rows.append({'epoch_1based':trainer.current_epoch+1,**module.get_last_validation_metrics()})

class StopAfter(pl.Callback):
    def __init__(self,epochs): self.epochs=epochs
    def on_train_epoch_end(self,trainer,module):
        if trainer.current_epoch+1>=self.epochs: trainer.should_stop=True

def load_trainable(module,state):
    expected={n for n,p in module.named_parameters() if p.requires_grad}
    if set(state)!=expected: raise ValueError(f'Checkpoint key mismatch: missing {sorted(expected-set(state))}; extra {sorted(set(state)-expected)}')
    for n,t in state.items():
        if not torch.isfinite(t).all(): raise ValueError(f'Non-finite checkpoint tensor: {n}')
    result=module.load_state_dict(state,strict=False)
    if result.unexpected_keys or any(not n.startswith('model.encoder.') for n in result.missing_keys): raise ValueError('Unexpected missing model state')

def choose_duration(rows,config):
    start=config['selection_start_epoch'];df=pd.DataFrame(rows)
    def col(n): return n if n in df else 'val_'+n
    rmse,pcc,rho,ratio=[col(x) for x in ['rmse','pearson_corr','spearman_corr','pred_target_std_ratio']]
    pool=df.loc[(df.epoch_1based>=start)&np.isfinite(df[rmse])].copy()
    if pool.empty: raise ValueError('No eligible validation epoch')
    good=pool[ratio]>=config['min_val_pred_std_ratio'];guard=bool(good.any())
    if guard: pool=pool.loc[good]
    pool=pool.loc[pool[rmse]<=pool[rmse].min()*(1+config['val_rmse_safe_fraction'])].copy()
    pool['score']=(pool[pcc].fillna(-1)+pool[rho].fillna(-1))/2
    row=pool.sort_values(['score',rmse,'epoch_1based'],ascending=[False,True,True]).iloc[0]
    selected=int(row.epoch_1based)
    return {'selected_epoch':selected,'refit_epochs':min(config['max_refit_epochs'],max(config['min_refit_epochs'],selected)),
            'validation_rmse':float(row[rmse]),'anti_collapse_guard_used':guard}
