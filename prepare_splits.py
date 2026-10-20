"""Generate and audit deterministic ordinary or low-homology five-fold manifests."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from diffcl.common import input_keys, Relations, validate_frame, audit_partitions, sha256, write_json

def prepare(data,protocol,output,homology=None,seed=42):
    frame=pd.read_csv(data);validate_frame(frame)
    if list(frame.record_id)!=list(range(len(frame))): raise ValueError('record_id must be contiguous and ordered')
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    strict=protocol=='lh30'
    relation=Relations(homology) if strict else None
    if relation: relation.require_coverage(frame)
    group=np.array(frame.PDB.astype(str).str.upper() if strict else input_keys(frame))
    unique=np.unique(group)
    if len(unique)<5: raise ValueError('At least five split groups are required')
    manifest={'schema_version':1,'protocol':protocol,'seed':seed,'data_sha256':sha256(data),
              'n_records':len(frame),'split_origin':'release-generated deterministic reconstruction',
              'outer_group':'PDB' if strict else 'identical four-sequence input',
              'inner_group':'PDB' if strict else 'identical four-sequence input','inner_validation_fraction':0.10,
              'inner_homology_purge':False,'folds':[]}
    if strict: manifest['homology_sha256']={n:sha256(Path(homology)/n) for n in ['sequences.fasta','all_vs_all.tsv']}
    for fold,(dev_groups,test_groups) in enumerate(KFold(5,shuffle=True,random_state=seed).split(unique),1):
        dev=frame.loc[np.isin(group,unique[dev_groups])];test=frame.loc[np.isin(group,unique[test_groups])]
        purged=[]
        if strict:
            forbidden=relation.forbidden_pdbs(dev,test)
            remove=dev.PDB.astype(str).str.upper().isin(forbidden)
            purged=dev.loc[remove,'record_id'].astype(int).tolist();dev=dev.loc[~remove]
        dev_group=group[dev.index]
        train_group,val_group=train_test_split(np.unique(dev_group),test_size=.10,random_state=seed+fold)
        train=dev.loc[np.isin(dev_group,train_group)];val=dev.loc[np.isin(dev_group,val_group)]
        counts=audit_partitions(train,val,dev,test,relation)
        f={'fold':fold,'counts':counts,'purged':purged}
        for n,d in [('inner_train',train),('val',val),('refit_train',dev),('test',test)]: f[n]=d.record_id.astype(int).tolist()
        manifest['folds'].append(f)
    tests=[i for f in manifest['folds'] for i in f['test']]
    if sorted(tests)!=list(range(len(frame))): raise AssertionError('Test records must occur exactly once')
    write_json(output/'manifest.json',manifest)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--protocol',choices=['ordinary','lh30'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--homology');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();m=prepare(a.data,a.protocol,a.output,a.homology,a.seed);print([f['counts'] for f in m['folds']])
