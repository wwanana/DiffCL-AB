"""Check deposited assets, data consistency, and checkpoint integrity on CPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from diffcl.common import sha256,write_json,input_keys,columns,encoded_depth,mutation_count,validate_frame,audit_partitions,Relations
from scripts.audit_cases import audit as audit_cases

def audit(output):
    import torch
    hashes=json.loads(Path('data/SHA256SUMS.json').read_text())
    for name,want in hashes.items():
        if sha256(name)!=want: raise ValueError(f'Checksum mismatch: {name}')
    core=json.loads(Path('checkpoints/source_sha256.json').read_text())
    for name,want in core.items():
        if sha256(name)!=want: raise ValueError(f'Core source differs from the checkpoint source: {name}')
    datasets={n:pd.read_csv(f'data/processed/{n}.csv') for n in ['AB645','S1131','AB1101']}
    assert {n:len(d) for n,d in datasets.items()}=={'AB645':645,'S1131':1130,'AB1101':1100}
    for n,d in datasets.items():validate_frame(d,n)
    def key(d):return Counter(zip(input_keys(d),d.PDB.str.upper(),d.ddG.astype(float)))
    frames={p:[pd.read_csv(f'data/protocols/{p}/{n}.csv') for n in ['inner_train','val','refit_train','test']] for p in ['mutd','mutd_lh30']}
    relation=Relations('data/homology/AB1101')
    counts={p:audit_partitions(*f,relation if p=='mutd_lh30' else None) for p,f in frames.items()}
    _,_,single,multi=frames['mutd']
    assert key(single)==key(datasets['AB645'])
    assert key(single)+key(multi)==key(datasets['AB1101'])
    assert key(multi)==key(frames['mutd_lh30'][-1])
    forbidden=relation.forbidden_pdbs(single,multi)
    assert key(single.loc[~single.PDB.str.upper().isin(forbidden)])==key(frames['mutd_lh30'][2])
    depth=np.array([encoded_depth(r,columns(multi)) for _,r in multi.iterrows()]);annotation=np.array([mutation_count(x) for x in multi.Mutation2])
    checkpoint=torch.load('checkpoints/mutd_seed42.pt',map_location='cpu',weights_only=True)
    state=checkpoint['state_dict'];assert len(state)==137
    assert all(torch.isfinite(t).all() for t in state.values())
    statistics=json.loads(Path('checkpoints/mutd_seed42.statistics.json').read_text())
    assert statistics['training_data_sha256']==sha256('data/protocols/mutd/refit_train.csv')
    np.testing.assert_allclose([statistics['mean'],statistics['std']],[single.ddG.mean(),single.ddG.std(ddof=1)],atol=1e-12,rtol=0)
    report={'status':'passed','verified_asset_count':len(hashes),'records':{n:len(d) for n,d in datasets.items()},'protocols':counts,
            'checkpoint_tensors':len(state),'checkpoint_parameters':sum(t.numel() for t in state.values()),
            'mutation_depth':{'test_records':len(multi),'annotation_multiple':int((annotation>=2).sum()),
              'encoded_multiple':int((depth>=2).sum()),'annotation_encoded_disagreement':int((depth!=annotation).sum())},
            'cases':audit_cases(),'cuda_available':torch.cuda.is_available()}
    write_json(output,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/release_audit.json');a=p.parse_args();print(audit(a.output))
