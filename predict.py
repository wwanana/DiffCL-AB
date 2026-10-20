"""Predict ΔΔG from a deposited downstream checkpoint and a local frozen ESM2."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from diffcl.common import validate_frame, sha256, write_json, metrics

def predict(args):
    import torch
    from diffcl.training import ReleaseLitModel,DataBundle,load_trainable
    from lit_model import MODEL_BATCH_KEYS,MUTATION_MASK_KEYS
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is required for the deposited Mamba checkpoint')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=dict(saved['config']);cfg.update(model_locate=str(Path(args.backbone).resolve()),local_files_only=True,
          num_workers=0,use_contrastive=False,allow_mamba_fallback=False,eval_batch_size=args.batch_size)
    statistics=saved.get('target_statistics')
    if statistics is None:
        statistics=json.loads(Path(args.statistics or str(Path(args.checkpoint).with_suffix('.statistics.json'))).read_text())
    frame=pd.read_csv(args.input);labeled='ddG' in frame
    if not labeled: frame['ddG']=0.
    validate_frame(frame,'inference input')
    bundle=DataBundle(cfg,None,target_statistics=(statistics['mean'],statistics['std']))
    net=ReleaseLitModel(cfg);load_trainable(net,saved['state_dict']);net.to(args.device).eval()
    rows=[]
    with torch.inference_mode():
        for batch in bundle.loader(frame,'inference'):
            inputs={k:batch[k].to(args.device) for k in MODEL_BATCH_KEYS+MUTATION_MASK_KEYS}
            raw=net.model(**inputs)*bundle.std+bundle.mean
            for idx,pred,pdb,mutation in zip(batch['source_indices'],raw.cpu(),batch['pdb_ids'],batch['mutations']):
                rows.append(dict(source_index=int(idx),pdb_id=pdb,mutation=mutation,y_pred=float(pred)))
    output=pd.DataFrame(rows)
    if labeled: output['y_true']=frame.ddG.to_numpy()
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True);output.to_csv(destination,index=False)
    report={'checkpoint_sha256':sha256(args.checkpoint),'input_sha256':sha256(args.input),'n':len(output),'target_statistics':statistics}
    if labeled: report['metrics']=metrics(output.y_true,output.y_pred)
    if args.compare:
        reference=pd.read_csv(args.compare)
        if reference.source_index.duplicated().any() or output.source_index.duplicated().any(): raise ValueError('Duplicated comparison identifiers')
        if set(reference.source_index)!=set(output.source_index): raise ValueError('Prediction comparison row mismatch')
        reference=reference.set_index('source_index').loc[output.source_index]
        if list(reference.pdb_id)!=list(output.pdb_id): raise ValueError('Prediction comparison complex mismatch')
        error=float(np.max(np.abs(output.y_pred.to_numpy()-reference.y_pred.to_numpy())))
        report['comparison_max_abs_error']=error;report['comparison_tolerance']=args.atol
        if labeled and not np.allclose(output.y_true,reference.y_true,atol=1e-6,rtol=0): raise ValueError('Prediction comparison target mismatch')
        write_json(destination.with_suffix('.json'),report)
        if error>args.atol: raise ValueError(f'Checkpoint predictions differ by {error:.6g}; tolerance {args.atol}')
    write_json(destination.with_suffix('.json'),report)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default='checkpoints/mutd_seed42.pt');p.add_argument('--backbone',default='models/esm2_650m')
    p.add_argument('--statistics');p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--compare');p.add_argument('--atol',type=float,default=1e-4);a=p.parse_args();predict(a)
