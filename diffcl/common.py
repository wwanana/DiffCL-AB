"""Data contracts and evaluation shared by every release experiment."""
from __future__ import annotations
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

AB = [('antibody_light_seq', 'antibody_heavy_seq'), ('antigen_a_seq', 'antigen_b_seq'),
      ('antibody_light_seq_mut', 'antibody_heavy_seq_mut'), ('antigen_a_seq_mut', 'antigen_b_seq_mut')]
PPI = [('a',), ('b',), ('a_mut',), ('b_mut',)]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def clean_sequence(value):
    return '' if pd.isna(value) else re.sub('[^A-Z]', '', str(value).upper())

def columns(frame):
    for schema in (AB, PPI):
        if all(c in frame for cs in schema for c in cs): return schema
    raise ValueError('Expected AB-Bind component columns or a, b, a_mut, b_mut')

def components(row, schema):
    return tuple(tuple(clean_sequence(row[c]) for c in cs) for cs in schema)

def states(row, schema):
    return tuple(''.join(cs) for cs in components(row, schema))

def input_keys(frame):
    schema = columns(frame)
    return [hashlib.sha256(json.dumps(states(r, schema)).encode()).hexdigest() for _, r in frame.iterrows()]

def mutation_count(value):
    return len(re.findall(r'[A-Za-z0-9]+:[A-Z]\d+[A-Za-z]?[A-Z]', str(value)))

def encoded_depth(row, schema):
    c = components(row, schema)
    return sum(a != b for i, j in ((0, 2), (1, 3)) for w, m in zip(c[i], c[j]) for a, b in zip(w, m))

def validate_frame(frame, context='data'):
    schema = columns(frame)
    if frame.empty: raise ValueError(f'{context}: empty dataset')
    if 'PDB' not in frame or 'ddG' not in frame: raise ValueError(f'{context}: missing PDB/ddG')
    if not np.isfinite(pd.to_numeric(frame.ddG, errors='coerce')).all(): raise ValueError(f'{context}: non-finite labels')
    for idx, row in frame.iterrows():
        if pd.isna(row.PDB) or not str(row.PDB).strip(): raise ValueError(f'{context}: empty PDB at {idx}')
        c = components(row, schema)
        if any(not ''.join(x) for x in c): raise ValueError(f'{context}: empty partner at {idx}')
        if any(len(a) != len(b) for i,j in ((0,2),(1,3)) for a,b in zip(c[i],c[j])):
            raise ValueError(f'{context}: unequal WT/MT component lengths at {idx}')
        if encoded_depth(row, schema) == 0: raise ValueError(f'{context}: unchanged WT/MT at {idx}')
    return schema

def metrics(target, predicted):
    y, p = np.asarray(target, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y): raise ValueError('Expected equal, nonempty vectors')
    if not np.isfinite(y).all() or not np.isfinite(p).all(): raise ValueError('Non-finite prediction or target')
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None
    sse = float(np.sum((p-y)**2)); sst = float(np.sum((y-y.mean())**2))
    return dict(n=len(y), rmse=float(np.sqrt(sse/len(y))), mae=float(np.abs(p-y).mean()),
                r2=1-sse/sst if sst>0 else None, pcc=corr(y,p), spearman=corr(rankdata(y),rankdata(p)))

class Relations:
    """Undirected qualifying direct hits; deliberately no transitive closure."""
    def __init__(self, directory, identity=.30, coverage=.80):
        self.directory = Path(directory); self.sequences = {}; current = None
        for line in (self.directory/'sequences.fasta').read_text().splitlines():
            if line.startswith('>'): current=line[1:].split()[0]; self.sequences[current]=''
            elif line.strip():
                if current is None: raise ValueError('Malformed FASTA')
                self.sequences[current] += line.strip()
        self.reverse = {v:k for k,v in self.sequences.items()}
        self.adj = defaultdict(set)
        with (self.directory/'all_vs_all.tsv').open() as f:
            for n,line in enumerate(f,1):
                parts=line.rstrip().split('\t')
                if len(parts)!=5: raise ValueError(f'Expected five MMseqs columns, line {n}')
                a,b=parts[:2]
                if a not in self.sequences or b not in self.sequences: raise ValueError('Unknown MMseqs sequence identifier')
                v=[float(x) for x in parts[2:]]
                v=[x/100 if x>1 else x for x in v]
                if not all(np.isfinite(v)) or any(x<0 or x>1 for x in v): raise ValueError('Invalid MMseqs fraction')
                if v[0]>=identity and min(v[1:])>=coverage:
                    self.adj[a].add(b); self.adj[b].add(a)
    def require_coverage(self, frame):
        schema=columns(frame)
        seq={s for _,r in frame.iterrows() for c in components(r,schema) for s in c if s}
        missing=seq-self.reverse.keys()
        if missing: raise ValueError(f'Homology database missing {len(missing)} component sequences; rebuild it')
    def related(self,a,b):
        if not a or not b: return False
        if a==b: return True
        return self.reverse.get(b) in self.adj.get(self.reverse.get(a),set())
    def forbidden_pdbs(self, candidate, heldout):
        self.require_coverage(candidate); self.require_coverage(heldout)
        schema=columns(heldout)
        forbidden={s for _,r in heldout.iterrows() for c in components(r,schema) for s in c if s}
        expanded=set(forbidden)
        for s in forbidden:
            expanded.update(self.sequences[n] for n in self.adj[self.reverse[s]])
        forbidden_pdb=set(heldout.PDB.astype(str).str.upper())
        schema=columns(candidate)
        for _,r in candidate.iterrows():
            if any(s in expanded for c in components(r,schema) for s in c if s): forbidden_pdb.add(str(r.PDB).upper())
        return forbidden_pdb

def audit_partitions(train, val, refit, test, relations=None):
    for name,df in [('train',train),('validation',val),('refit',refit),('test',test)]: validate_frame(df,name)
    sets=[set(input_keys(d)) for d in [train,val,refit,test]]
    if sets[0]&sets[1] or sets[2]&sets[3]: raise ValueError('Identical model input crosses a partition boundary')
    from collections import Counter
    def records(d):
        return Counter(zip(input_keys(d),d.PDB.astype(str).str.upper(),d.ddG.astype(float)))
    if records(train)+records(val)!=records(refit): raise ValueError('Refit must contain exactly train + validation records, including labels')
    if relations is not None:
        for a,b in [(refit,test)]:
            if set(a.PDB.astype(str).str.upper()) & relations.forbidden_pdbs(a,b):
                raise ValueError('Complex or qualifying sequence relation crosses a strict partition')
        if set(train.PDB.astype(str).str.upper()) & set(val.PDB.astype(str).str.upper()):
            raise ValueError('Complex crosses the strict inner validation partition')
    return {'train':len(train),'validation':len(val),'refit':len(refit),'test':len(test)}
