"""Validate external baseline predictions against an exported test partition."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from diffcl.common import metrics,sha256,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--test',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--method',required=True);p.add_argument('--implementation',required=True,help='Upstream repository URL and exact commit/version')
    p.add_argument('--fold',type=int,default=0);p.add_argument('--output',required=True)
    a=p.parse_args();truth=pd.read_csv(a.test).set_index('record_id');pred=pd.read_csv(a.predictions)
    if not {'source_index','y_pred'}<=set(pred):raise ValueError('Expected source_index and y_pred in kcal/mol')
    if pred.source_index.duplicated().any() or set(pred.source_index)!=set(truth.index):raise ValueError('Missing, extra, or repeated test row')
    truth=truth.loc[pred.source_index]
    if 'y_true' in pred and not np.allclose(pred.y_true,truth.ddG,atol=1e-6,rtol=0):raise ValueError('External target differs from source')
    pred['y_true']=truth.ddG.to_numpy();pred['pdb_id']=truth.PDB.to_numpy();pred['fold']=a.fold
    summary=metrics(pred.y_true,pred.y_pred)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);pred.to_csv(out/'predictions.csv',index=False)
    write_json(out/'metrics.json',summary)
    write_json(out/'provenance.json',{'method':a.method,'implementation':a.implementation,'source_prediction_sha256':sha256(a.predictions),'test_sha256':sha256(a.test)})
