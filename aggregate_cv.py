"""Audit complete out-of-fold predictions and report pooled and fold-level metrics."""
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from diffcl.common import metrics,write_json,sha256

def aggregate(data,files,output):
    source=pd.read_csv(data).set_index('record_id',drop=False)
    paths=sorted(Path(p) for p in files)
    if len(paths)!=5: raise ValueError('Supply exactly five prediction files for one method/seed/protocol')
    frames=[pd.read_csv(p) for p in paths]
    if any(len(x.fold.unique())!=1 for x in frames): raise ValueError('Each file must represent one fold')
    df=pd.concat(frames,ignore_index=True)
    if set(df.fold)!={1,2,3,4,5}: raise ValueError('Expected folds 1 through 5')
    if df.source_index.duplicated().any() or set(df.source_index)!=set(source.index): raise ValueError('Every dataset row must occur exactly once')
    df=df.sort_values('source_index');expected=source.loc[df.source_index]
    if not np.allclose(df.y_true,expected.ddG,atol=1e-6,rtol=0): raise ValueError('OOF labels differ from source labels')
    if list(df.pdb_id.str.upper())!=list(expected.PDB.str.upper()): raise ValueError('OOF complex identifiers differ')
    by_fold=[{'fold':int(f),**metrics(x.y_true,x.y_pred)} for f,x in df.groupby('fold')]
    report={'pooled':metrics(df.y_true,df.y_pred),'by_fold':by_fold,'fold_mean':{},'fold_sample_sd':{},'n_defined_folds':{},
            'data_sha256':sha256(data),'prediction_sha256':{str(p):sha256(p) for p in paths}}
    for k in ['rmse','mae','r2','pcc','spearman']:
        values=[r[k] for r in by_fold if r[k] is not None]
        report['n_defined_folds'][k]=len(values)
        report['fold_mean'][k]=float(np.mean(values)) if values else None
        report['fold_sample_sd'][k]=float(np.std(values,ddof=1)) if len(values)>1 else None
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'oof_predictions.csv',index=False);pd.DataFrame(by_fold).to_csv(out/'fold_metrics.csv',index=False)
    write_json(out/'metrics.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(aggregate(a.data,glob.glob(a.predictions),a.output)['pooled'])
