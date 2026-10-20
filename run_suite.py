"""One command for the complete DiffCL-AB evaluation and ablation matrix."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from diffcl.common import write_json

def commands(experiment,datasets,output,backbone,config,seeds):
    for seed in seeds:
        jobs=[]
        if experiment in ('all','ordinary','lh30'):
            protocols=['ordinary','lh30'] if experiment=='all' else [experiment]
            jobs += [(d,p,'full',f) for d in datasets for p in protocols for f in range(1,6)]
        if experiment in ('all','ablation'):
            jobs += [(d,'ordinary',v,f) for d in datasets for v in ['cnn','no_cl1','no_cl2','no_cl','simple_diff'] for f in range(1,6)]
        if experiment in ('all','mutd','mutd_lh30') and 'AB1101' in datasets:
            jobs += [('AB1101',p,'full',1) for p in (['mutd','mutd_lh30'] if experiment=='all' else [experiment])]
        for d,p,v,f in jobs:
            out=Path(output)/d/p/v/f'seed{seed}'/f'fold{f}'
            yield [sys.executable,'run_experiment.py','--dataset',d,'--protocol',p,'--variant',v,'--fold',str(f),
                   '--seed',str(seed),'--backbone',backbone,'--config',config,'--output',str(out)]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',choices=['all','ordinary','lh30','ablation','mutd','mutd_lh30'],default='all')
    p.add_argument('--datasets',nargs='+',choices=['AB645','S1131','AB1101'],default=['AB645','S1131','AB1101'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42]);p.add_argument('--backbone',default='models/esm2_650m')
    p.add_argument('--config',default='configs/reference.json');p.add_argument('--output',default='runs')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--preflight',action='store_true')
    a=p.parse_args();jobs=list(commands(a.experiment,a.datasets,a.output,a.backbone,a.config,a.seeds))
    if not jobs: raise ValueError('No jobs selected; mutation-depth evaluation requires AB1101')
    write_json(Path(a.output)/'commands.json',jobs)
    print(f'{len(jobs)} experiment jobs; commands saved in {a.output}/commands.json')
    if not a.plan_only:
        for cmd in jobs: subprocess.run(cmd+(['--preflight'] if a.preflight else []),check=True)
