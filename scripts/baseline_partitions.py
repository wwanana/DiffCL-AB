"""Export identical evaluation partitions for independently installed comparison methods."""
import argparse
from pathlib import Path
from run_experiment import load_frames
from diffcl.common import write_json,sha256

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['AB645','S1131','AB1101'],required=True)
    p.add_argument('--protocol',choices=['ordinary','lh30','mutd','mutd_lh30'],required=True)
    p.add_argument('--fold',type=int,choices=range(1,6),default=1);p.add_argument('--output',required=True)
    a=p.parse_args();a.splits=None;a.homology='data/homology/S1131' if a.dataset=='S1131' else 'data/homology/AB1101'
    if a.protocol.startswith('mutd') and a.dataset!='AB1101':raise ValueError('Mutation-depth protocols require AB1101')
    frames,provenance=load_frames(a);out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in zip(['inner_train','val','refit_train','test'],frames):
        frame=frame.copy()
        if 'record_id' not in frame:frame.insert(0,'record_id',range(len(frame)))
        frame.to_csv(out/f'{name}.csv',index=False)
    provenance.update(dataset=a.dataset,protocol=a.protocol,fold=a.fold)
    write_json(out/'manifest.json',provenance)
