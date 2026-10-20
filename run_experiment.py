"""Run validation-based selection, fresh refitting, and one held-out evaluation."""
from __future__ import annotations
import argparse
import gc
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from diffcl.common import audit_partitions, Relations, sha256, write_json, metrics

def load_frames(args):
    if args.protocol in ('mutd','mutd_lh30'):
        root=Path('data/protocols')/args.protocol
        frames=[pd.read_csv(root/f'{n}.csv') for n in ['inner_train','val','refit_train','test']]
        provenance={'files':{n:sha256(root/f'{n}.csv') for n in ['inner_train','val','refit_train','test']},'origin':'deposited experiment partitions'}
    else:
        path=Path(args.splits or f'data/splits/{args.dataset}/{args.protocol}/manifest.json')
        manifest=json.loads(path.read_text());data=Path(f'data/processed/{args.dataset}.csv')
        if manifest['data_sha256']!=sha256(data) or manifest['protocol']!=args.protocol: raise ValueError('Split/data/protocol mismatch')
        if manifest.get('homology_sha256'):
            for n,h in manifest['homology_sha256'].items():
                if sha256(Path(args.homology)/n)!=h: raise ValueError('Split/homology version mismatch')
        f=next(x for x in manifest['folds'] if x['fold']==args.fold);dataframe=pd.read_csv(data).set_index('record_id',drop=False)
        frames=[dataframe.loc[f[n]].reset_index(drop=True) for n in ['inner_train','val','refit_train','test']]
        provenance={'manifest_sha256':sha256(path),'data_sha256':sha256(data),'origin':manifest['split_origin'],'fold':args.fold}
    relation=Relations(args.homology) if args.protocol in ('lh30','mutd_lh30') else None
    counts=audit_partitions(*frames,relation)
    # The CL mask uses 90%, independently of the outer 30% restriction.
    relation90=Relations(args.homology,identity=.90)
    for f in frames: relation90.require_coverage(f)
    provenance.update(counts=counts,homology_sha256={n:sha256(Path(args.homology)/n) for n in ['sequences.fasta','all_vs_all.tsv']})
    return frames,provenance

def prediction_frame(rows,fold):
    df=pd.DataFrame(rows).rename(columns={'target_raw':'y_true','pred_raw':'y_pred'})
    df=df[['source_index','pdb_id','mutation','y_true','y_pred']].sort_values('source_index').reset_index(drop=True)
    df.insert(0,'fold',fold);df['error']=df.y_pred-df.y_true;df['abs_error']=abs(df.error)
    return df

def run(args):
    frames,provenance=load_frames(args)
    cfg=json.loads(Path(args.config).read_text());selection=cfg.pop('selection')
    cfg.update(model_locate=str(Path(args.backbone).resolve()),variant=args.variant,seed=args.seed,num_workers=args.workers)
    provenance.update(protocol=args.protocol,dataset=args.dataset,variant=args.variant,seed=args.seed,config=cfg,selection=selection)
    if args.preflight:
        print(json.dumps({'status':'data and partitions valid','protocol':args.protocol,'dataset':args.dataset,'counts':provenance['counts']},indent=2));return provenance
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from diffcl.training import ReleaseLitModel, DataBundle, ValidationHistory, StopAfter, choose_duration
    if not torch.cuda.is_available(): raise RuntimeError('Formal experiments require CUDA; use pytest for the CPU functionality checks')
    if cfg.get('allow_mamba_fallback',False): raise ValueError('A debug fallback cannot be used in formal experiments')
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'{out}: choose an empty run directory')
    out.mkdir(parents=True,exist_ok=True)
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (FileNotFoundError,subprocess.CalledProcessError): commit=None
    source_files=sorted(list(Path('.').glob('*.py'))+list(Path('diffcl').glob('*.py'))+list(Path('scripts').glob('*.py')))
    provenance.update(git_commit=commit,source_sha256={str(p):sha256(p) for p in source_files},
        environment={'python':platform.python_version(),'torch':str(torch.__version__),'lightning':pl.__version__,
                     'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)})
    from importlib.metadata import version,PackageNotFoundError
    provenance['environment']['packages']={}
    for package in ['torch','mamba-ssm','transformers','pytorch-lightning','numpy','pandas','tokenizers']:
        try: provenance['environment']['packages'][package]=version(package)
        except PackageNotFoundError: provenance['environment']['packages'][package]=None
    backbone=Path(args.backbone)
    provenance['backbone_files']={p.name:sha256(p) for p in sorted(backbone.iterdir()) if p.is_file() and p.suffix in ('.json','.txt','.bin','.safetensors')}
    write_json(out/'run_manifest.json',provenance)
    torch.backends.cudnn.benchmark=False
    def trainer(stage,callbacks,validation):
        return pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_epochs=selection['horizon_epochs'],
            gradient_clip_val=1.0,deterministic=True,logger=CSVLogger(str(out),name=stage),callbacks=callbacks,
            enable_checkpointing=False,num_sanity_val_steps=0,limit_val_batches=1.0 if validation else 0,
            log_every_n_steps=1,enable_model_summary=False)
    train,val,refit,test=frames
    pl.seed_everything(args.seed,workers=True)
    cfg['training_stage']='selection';bundle=DataBundle(cfg,train,args.homology);history=ValidationHistory()
    net=ReleaseLitModel(cfg);selection_trainer=trainer('selection',[history],True)
    selection_trainer.fit(net,bundle.loader(train,'train',True),bundle.loader(val,'val'))
    pd.DataFrame(history.rows).to_csv(out/'validation_history.csv',index=False)
    duration=choose_duration(history.rows,selection);write_json(out/'selection.json',duration)
    del net,selection_trainer,bundle;gc.collect();torch.cuda.empty_cache()
    pl.seed_everything(args.seed,workers=True)
    cfg['training_stage']='refit';bundle=DataBundle(cfg,refit,args.homology)
    net=ReleaseLitModel(cfg);final_trainer=trainer('refit',[StopAfter(duration['refit_epochs'])],False)
    final_trainer.fit(net,bundle.loader(refit,'refit',True))
    stats={'mean':bundle.mean,'std':bundle.std,'ddof':1,'n_train':len(refit)}
    write_json(out/'target_statistics.json',stats)
    weights={n:p.detach().cpu() for n,p in net.named_parameters() if p.requires_grad}
    torch.save({'format':'diffcl-ab-trainable-v1','state_dict':weights,'config':cfg,'target_statistics':stats},out/'trainable_state.pt')
    final_trainer.test(net,dataloaders=bundle.loader(test,'test'),verbose=False)
    predictions=prediction_frame(net.get_last_test_predictions(),args.fold if args.protocol in ('ordinary','lh30') else 0)
    expected=test.record_id.to_numpy() if 'record_id' in test else np.arange(len(test))
    if list(predictions.source_index)!=sorted(expected): raise ValueError('Incomplete or duplicated test predictions')
    source=test.assign(source_index=expected).set_index('source_index').loc[predictions.source_index]
    if not np.allclose(predictions.y_true,source.ddG,atol=1e-6,rtol=0): raise ValueError('Test labels changed in prediction export')
    predictions.to_csv(out/'predictions.csv',index=False)
    write_json(out/'metrics.json',metrics(predictions.y_true,predictions.y_pred))
    write_json(out/'completed.json',{'prediction_sha256':sha256(out/'predictions.csv'),'checkpoint_sha256':sha256(out/'trainable_state.pt')})
    return provenance

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['AB645','AB1101','S1131'],default='AB1101')
    p.add_argument('--protocol',choices=['ordinary','lh30','mutd','mutd_lh30'],required=True)
    p.add_argument('--fold',type=int,choices=range(1,6),default=1);p.add_argument('--splits')
    p.add_argument('--homology');p.add_argument('--backbone',default='models/esm2_650m')
    p.add_argument('--config',default='configs/reference.json');p.add_argument('--output',required=True)
    p.add_argument('--variant',choices=['full','cnn','no_cl1','no_cl2','no_cl','simple_diff'],default='full')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,default=0);p.add_argument('--preflight',action='store_true')
    return p

if __name__=='__main__':
    a=parser().parse_args()
    if a.protocol.startswith('mutd') and a.dataset!='AB1101': raise ValueError('Mutation-depth protocols use AB1101')
    if not a.homology: a.homology='data/homology/S1131' if a.dataset=='S1131' else 'data/homology/AB1101'
    run(a)
