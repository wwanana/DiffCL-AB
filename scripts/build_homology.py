"""Compute direct component-sequence relationships with MMseqs2."""
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from diffcl.common import columns, components, sha256, write_json, validate_frame

def build(csv,output,mmseqs='mmseqs',threads=4):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'all_vs_all.tsv').exists(): raise FileExistsError('Choose a new homology output directory')
    df=pd.read_csv(csv);schema=validate_frame(df)
    seqs=sorted({s for _,r in df.iterrows() for cs in components(r,schema) for s in cs if s})
    fasta=output/'sequences.fasta'
    fasta.write_text(''.join(f'>s{i:07d}\n{s}\n' for i,s in enumerate(seqs)))
    command=[str(mmseqs),'easy-search',str(fasta),str(fasta),str(output/'all_vs_all.tsv'),str(output/'tmp'),
             '--min-seq-id','0.30','-c','0.80','--cov-mode','0','-s','7.5',
             '--max-seqs',str(max(10000,len(seqs))),'--threads',str(threads),
             '--format-output','query,target,fident,qcov,tcov']
    version=subprocess.check_output([str(mmseqs),'version'],text=True).strip()
    with (output/'search.log').open('w') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(output/'provenance.json',{'version':version,'command':command,'input_sha256':sha256(csv),
               'fasta_sha256':sha256(fasta),'hits_sha256':sha256(output/'all_vs_all.tsv'),
               'relation':'identity >= threshold and qcov >= 0.80 and tcov >= 0.80; direct edges, no transitive closure'})
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mmseqs',default='mmseqs');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();print(build(a.data,a.output,a.mmseqs,a.threads))
