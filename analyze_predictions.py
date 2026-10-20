"""Recompute held-out metrics and diagnostic subsets without selecting model settings."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from diffcl.common import metrics,columns,encoded_depth,mutation_count,write_json,sha256

def analyze(predictions,test,train,output):
    p=pd.read_csv(predictions).sort_values('source_index').reset_index(drop=True)
    t=pd.read_csv(test);tr=pd.read_csv(train)
    ids=t.record_id.to_numpy() if 'record_id' in t else np.arange(len(t))
    t=t.assign(source_index=ids).set_index('source_index').loc[p.source_index]
    if len(p)!=len(t) or set(p.source_index)!=set(ids) or p.source_index.duplicated().any(): raise ValueError('Test row mismatch')
    if list(p.pdb_id.str.upper())!=list(t.PDB.str.upper()) or not np.allclose(p.y_true,t.ddG,atol=1e-6,rtol=0): raise ValueError('Test identity/label mismatch')
    p['represented_complex']=p.pdb_id.str.upper().isin(set(tr.PDB.str.upper()))
    p['assigned_plus8']=np.isclose(p.y_true,8.,atol=1e-6,rtol=0)
    p['encoded_depth']=[encoded_depth(r,columns(t)) for _,r in t.iterrows()]
    if 'Mutation2' in t: p['annotation_depth']=[mutation_count(x) for x in t.Mutation2]
    report={'all':metrics(p.y_true,p.y_pred),'source_prediction_sha256':sha256(predictions)}
    subsets={'quantitative':~p.assigned_plus8,'represented_complex':p.represented_complex,'unrepresented_complex':~p.represented_complex,
             'quantitative_represented':~p.assigned_plus8&p.represented_complex,'quantitative_unrepresented':~p.assigned_plus8&~p.represented_complex,
             'encoded_multiple':p.encoded_depth>=2}
    for n,mask in subsets.items():
        s=p.loc[mask];report[n]=metrics(s.y_true,s.y_pred) if len(s) else {'n':0}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    p.to_csv(out/'annotated_predictions.csv',index=False);write_json(out/'metrics.json',report)
    for c in ['pdb_id','encoded_depth']+(['annotation_depth'] if 'annotation_depth' in p else []):
        pd.DataFrame([{c:key,**metrics(s.y_true,s.y_pred)} for key,s in p.groupby(c)]).to_csv(out/f'by_{c}.csv',index=False)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',default='results/archived/mutd_predictions.csv')
    p.add_argument('--test',default='data/protocols/mutd/test.csv');p.add_argument('--train',default='data/protocols/mutd/refit_train.csv')
    p.add_argument('--output',default='runs/mutd_analysis');a=p.parse_args();print(analyze(a.predictions,a.test,a.train,a.output))
