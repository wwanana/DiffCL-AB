"""Verify case identities, predictions, and shortest WT heavy-atom separations."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from diffcl.common import write_json

AA=dict(ALA='A',ARG='R',ASN='N',ASP='D',CYS='C',GLN='Q',GLU='E',GLY='G',HIS='H',ILE='I',LEU='L',LYS='K',MET='M',PHE='F',PRO='P',SER='S',THR='T',TRP='W',TYR='Y',VAL='V')

def atoms(path):
    groups={}
    for s in Path(path).read_text().splitlines():
        if s.startswith('ENDMDL'):break
        if s.startswith('ATOM  ') and s[16] in ' A' and s[76:78].strip()!='H':
            key=(s[21],int(s[22:26]),s[26].strip())
            groups.setdefault(key,[]).append((s[12:16].strip(),AA[s[17:20]],np.array([float(s[30:38]),float(s[38:46]),float(s[46:54])])))
    return groups

def audit():
    data=Path('data/cases');cases=pd.read_csv(data/'case_predictions.csv')
    pred=pd.read_csv('results/archived/mutd_predictions.csv').set_index('source_index')
    coordinates={p:atoms(data/'structures'/f'{p}.pdb') for p in cases.pdb_id}
    def site(p,annotation):
        chain,wt,pos,_=re.fullmatch(r'([A-Za-z]):([A-Z])(\d+)([A-Z]?)',annotation).groups()
        a=coordinates[p][(chain,int(pos),'')]
        assert all(x[1]==wt for x in a),f'WT residue mismatch: {p} {annotation}'
        return a
    for _,r in cases.iterrows():
        expected=pred.loc[r.source_index]
        assert expected.pdb_id==r.pdb_id and expected.mutation_annotation==r.mutation
        np.testing.assert_allclose([r.y_true,r.y_pred],[expected.y_true,expected.y_pred],rtol=0,atol=1e-12)
        for m in r.mutation.split(','):site(r.pdb_id,m)
    distances=pd.read_csv(data/'wt_contact_distances.csv');errors=[]
    for _,r in distances.iterrows():
        a=site(r.PDB,r.mutated_WT_site);b=site(r.PDB,r.opposite_chain_residue)
        d=np.linalg.norm(np.stack([x[2] for x in a])[:,None,:]-np.stack([x[2] for x in b])[None,:,:],axis=2)
        i,j=np.unravel_index(d.argmin(),d.shape)
        assert (a[i][0],b[j][0])==(r.site_atom,r.opposite_atom)
        errors.append(abs(float(d[i,j])-r.distance_A))
    assert max(errors)<1e-6
    return {'cases':len(cases),'verified_distances':len(distances),'maximum_rounding_error_A':max(errors),
            'coordinates':'wild type only','selection':'post hoc; represented parent complexes; not independent validation'}

if __name__=='__main__':
    result=audit();write_json('runs/case_audit.json',result);print(result)
