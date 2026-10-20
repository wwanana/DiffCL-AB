"""Rebuild model-valid data from the deposited source CSVs without changing labels."""
import argparse
from collections import defaultdict, deque
from pathlib import Path
import pandas as pd
from diffcl.common import AB, columns, states, input_keys, mutation_count, encoded_depth, validate_frame, write_json, sha256

def build(source, output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    ab=pd.read_csv(source/'AB645.csv');mixed=pd.read_csv(source/'AB1101.csv');ppi=pd.read_csv(source/'S1131.csv')
    buckets=defaultdict(deque)
    def key(row,mutation): return (str(row.PDB).upper(),str(row[mutation]).replace(',','').replace(' ',''),float(row.ddG))
    for _,r in ab.iterrows(): buckets[key(r,'Mutation')].append(r)
    changed=[]
    for i,r in mixed.iterrows():
        if mutation_count(r.Mutation2)!=1: continue
        k=key(r,'Mutation2')
        if not buckets[k]: raise ValueError(f'Unmatched AB1101 single-substitution row {i}')
        donor=buckets[k].popleft()
        if states(r,AB)!=states(donor,AB): changed.append(i)
        for cs in AB:
            for c in cs: mixed.at[i,c]=donor[c]
    if any(buckets.values()): raise ValueError('Unconsumed AB645 rows')
    ppi_schema=columns(ppi)
    keep=[encoded_depth(r,ppi_schema)>0 for _,r in ppi.iterrows()]
    excluded=ppi.loc[[not x for x in keep]].copy();excluded.insert(0,'source_row',excluded.index)
    excluded.to_csv(output/'S1131_excluded.csv',index=False)
    report={'source_sha256':{n:sha256(source/f'{n}.csv') for n in ['AB645','AB1101','S1131']},
            'AB1101_sequence_reconciled_source_rows':changed,'S1131_excluded_unchanged_source_rows':list(excluded.source_row)}
    for name,frame in [('AB645',ab),('AB1101',mixed),('S1131',ppi.loc[keep])]:
        frame=frame.copy();frame.insert(0,'source_row',frame.index);frame=frame.reset_index(drop=True)
        frame.insert(0,'record_id',range(len(frame)));validate_frame(frame,name)
        frame.to_csv(output/f'{name}.csv',index=False)
        keys=pd.Series(input_keys(frame));groups=frame.assign(input_key=keys).groupby('input_key')
        report[name]={'records':len(frame),'complexes':int(frame.PDB.nunique()),
                     'repeated_input_rows':int(keys.duplicated(keep=False).sum()),
                     'conflicting_input_groups':int((groups.ddG.nunique()>1).sum()),
                     'assigned_8_labels':int((frame.ddG==8).sum()) if name.startswith('AB') else 0}
    write_json(output/'preparation.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='data/source');p.add_argument('--output',default='data/processed')
    a=p.parse_args();print(build(a.source,a.output))
