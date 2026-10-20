"""Render actual wild-type PDB coordinates, with verified mutation identities.

All separations are shortest heavy-atom distances, not assigned hydrogen bonds.
No mutant coordinates or model attention maps are inferred or synthesized.
"""
from pathlib import Path
import argparse
import json
import math
import re
import csv
import numpy as np
import pymol
pymol.finish_launching(['pymol', '-cq'])
from pymol import cmd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO/'runs'/'figures'
STRUCTURES = REPO/'data'/'cases'/'structures'
OUT = ROOT/'render'
COLORS = {'heavy': [.18,.38,.58], 'light': [.18,.58,.61], 'antigen': [.70,.73,.77],
          'mutation': [.80,.26,.24], 'neighbor': [.84,.63,.28]}
AA1 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G',
           HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F', PRO='P', SER='S',
           THR='T', TRP='W', TYR='Y', VAL='V')
CASES = [
    {'pdb': '1N8Z', 'mutations': ['A:N30S','A:H91F','A:Y92W'], 'anchor': ('A',92),
     'title': 'Herceptin Fab–HER2', 'angle': 15},
    {'pdb': '1DQJ', 'mutations': ['B:W98A','C:Y20A'], 'anchor': ('B',98),
     'title': 'HYHEL-63–lysozyme', 'angle': -15},
    {'pdb': '1VFB', 'mutations': ['A:Y32A','C:Q121A'], 'anchor': ('A',32),
     'title': 'D1.3 Fv–lysozyme', 'angle': 20},
]


def read_atoms(pdb):
    atoms=[]
    for line in (STRUCTURES/f'{pdb}.pdb').read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if line.startswith('ATOM  ') and line[16] in ' A' and line[76:78].strip() != 'H':
            atoms.append({'chain':line[21], 'resi':int(line[22:26]), 'icode':line[26].strip(),
                          'resn':line[17:20], 'name':line[12:16].strip(),
                          'xyz':np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])})
    return atoms


def key(a):
    return a['chain'],a['resi'],a['icode']


def sel(k):
    return f'(chain {k[0]} and resi {k[1]}{k[2]})'


def audit(c):
    aa=read_atoms(c['pdb'])
    groups={}
    for a in aa:
        groups.setdefault(key(a), []).append(a)
    c['sites']=[]
    c['contacts']=[]
    for m in c['mutations']:
        chain,wt,pos,mt=re.fullmatch(r'([A-Za-z]):([A-Z])(\d+)([A-Z])',m).groups()
        k=(chain,int(pos),'')
        site=groups[k]
        if any(AA1.get(a['resn']) != wt for a in site):
            raise ValueError(f'WT identity mismatch {c["pdb"]}:{m}')
        side=[a for a in site if a['name'] not in {'N','CA','C','O'}]
        c['sites'].append({'mutation':m,'chain':chain,'resi':int(pos),'wt':wt,'mt':mt,
                           'label_xyz':np.mean([a['xyz'] for a in side],axis=0).tolist()})
        opposite=[(k2,vals) for k2,vals in groups.items()
                  if (k2[0] == 'C' if chain != 'C' else k2[0] in ('A','B'))]
        candidates=[]
        for k2,other in opposite:
            d=np.linalg.norm(np.stack([a['xyz'] for a in site])[:,None,:]-np.stack([a['xyz'] for a in other])[None,:,:],axis=2)
            i,j=np.unravel_index(np.argmin(d),d.shape)
            candidates.append({'site':m, 'site_key':list(k),'partner_key':list(k2),
                               'partner':f'{k2[0]}:{AA1[other[j]["resn"]]}{k2[1]}{k2[2]}',
                               'distance_A':float(d[i,j]),'site_atom':site[i]['name'],
                               'partner_atom':other[j]['name'],
                               'site_xyz':site[i]['xyz'].tolist(),'partner_xyz':other[j]['xyz'].tolist()})
        selected=sorted(candidates,key=lambda n:n['distance_A'])[:3]
        # Also retain opposite-side mutated residues, so pair labels have exact evidence.
        opposite_mutations={(x.split(':')[0],int(re.search(r'\d+',x).group()),'') for x in c['mutations']}
        for n in candidates:
            if tuple(n['partner_key']) in opposite_mutations and n['distance_A'] <= 5.0 and n not in selected:
                selected.append(n)
        c['contacts'].extend(selected)
    return aa,groups


def frame(c,groups):
    site=groups[(*c['anchor'],'')]
    xyz={a['name']:a['xyz'] for a in site}
    center=np.mean([xyz[k] for k in ('CG','CD1','CD2')],axis=0)
    near=next(n for n in c['contacts'] if tuple(n['site_key'][:2]) == c['anchor'])
    x=np.asarray(near['partner_xyz'])-center
    x/=np.linalg.norm(x)
    normal=np.cross(xyz['CD1']-xyz['CG'],xyz['CD2']-xyz['CG'])
    z=normal-x*np.dot(normal,x)
    z/=np.linalg.norm(z)
    y=np.cross(z,x)
    a=math.radians(c['angle'])
    y=math.cos(a)*y+math.sin(a)*z
    z=np.cross(x,y)
    r=np.stack([x,y,z])
    assert np.linalg.det(r)>0.999
    return r


def set_view(center,span,w,h):
    distance=span/(2*math.tan(math.radians(10)))
    cmd.viewport(w,h)
    cmd.set_view([1,0,0,0,1,0,0,0,1,0,0,-distance,*center,
                  max(1,distance-180),distance+180,1])
    return {'center':list(center),'span':float(span),'width':w,'height':h}


def project(xyz,r,v):
    x=r@np.asarray(xyz)-np.asarray(v['center'])
    return [v['width']/2+x[0]*v['height']/v['span'],v['height']/2-x[1]*v['height']/v['span']]


def render(c,preview):
    aa,groups=audit(c)
    r=frame(c,groups)
    cmd.reinitialize()
    cmd.load(str(STRUCTURES/f'{c["pdb"]}.pdb'),'complex')
    cmd.remove('not polymer.protein or hydro')
    cmd.remove('not alt "" and not alt A')
    cmd.alter('all','alt=""')
    cmd.alter_state(1,'all','(x,y,z)=tuple(r @ np.array([x,y,z]))',space={'r':r,'np':np})
    for name,color in COLORS.items():
        cmd.set_color(name,color)
    cmd.bg_color('white')
    cmd.hide('everything')
    for chain,color in [('A','light'),('B','heavy'),('C','antigen')]:
        cmd.color(color,f'chain {chain}')
        cmd.set('cartoon_color',color,f'chain {chain}')
    for setting,value in [('orthoscopic',1),('field_of_view',20),('ray_opaque_background',1),
                           ('antialias',2),('ambient',.48),('direct',.56),('reflect',.04),
                           ('specular',.14),('shininess',25),('ray_shadows',0),('depth_cue',0),
                           ('cartoon_fancy_helices',1),('cartoon_fancy_sheets',1),
                           ('cartoon_smooth_loops',1),('cartoon_loop_radius',.20),
                           ('stick_radius',.20),('sphere_scale',.24),('ray_trace_mode',1),
                           ('ray_trace_gain',.025)]:
        cmd.set(setting,value)
    sitekeys=[(s['chain'],s['resi'],'') for s in c['sites']]
    cmd.select('mut_sites',' or '.join(sel(k) for k in sitekeys))
    cmd.show('cartoon','all')
    cmd.show('sticks','mut_sites and (sidechain or name CA)')
    cmd.color('mutation','mut_sites')
    xy=np.stack([r@a['xyz'] for a in aa if a['name']=='CA'])
    lo,hi=xy.min(axis=0),xy.max(axis=0)
    center=(lo+hi)/2
    w,h=(520,430) if preview else (1600,1320)
    span=max((hi[1]-lo[1])*1.12,(hi[0]-lo[0])*h/w*1.12)
    view=set_view(center,span,w,h)
    cmd.png(str(OUT/f'{c["pdb"]}_overview.png'),w,h,ray=1)
    c['overview_view']=view
    for s in c['sites']:
        s['overview_point']=project(s['label_xyz'],r,view)

    cmd.hide('cartoon','all')
    cmd.select('local','byres (complex within 10.5 of mut_sites)')
    cmd.show('cartoon','local')
    cmd.set('cartoon_transparency',.24,'local')
    cmd.set('cartoon_transparency',.52,'chain C')
    neighbor_keys={tuple(n['partner_key']) for n in c['contacts']}
    cmd.select('near_res',' or '.join(sel(k) for k in neighbor_keys))
    cmd.color('neighbor','near_res')
    cmd.show('sticks','(mut_sites or near_res) and (sidechain or name CA)')
    cmd.color('mutation','mut_sites')
    # Draw one shortest contact per mutation; duplicate pairs drawn once.
    pairs=set()
    drawn=[]
    for s in c['sites']:
        candidates=[n for n in c['contacts'] if n['site']==s['mutation']]
        n=next((n for n in candidates if tuple(n['partner_key']) in sitekeys),candidates[0])
        pair=tuple(sorted((tuple(n['site_key']),tuple(n['partner_key']))))
        if pair in pairs:
            continue
        pairs.add(pair)
        drawn.append(n)
        name=f'contact_{len(drawn)}'
        cmd.distance(name,sel(n['site_key'])+f' and name {n["site_atom"]}',
                     sel(n['partner_key'])+f' and name {n["partner_atom"]}',mode=0)
        cmd.hide('labels',name)
        cmd.set('dash_color','gray45',name)
        cmd.set('dash_gap',.32,name)
        cmd.set('dash_length',.28,name)
        cmd.set('dash_width',1.6,name)
    c['drawn_contacts']=drawn
    pts=np.stack([r@a['xyz'] for k in sitekeys for a in groups[k]]+
                 [r@np.asarray(n['partner_xyz']) for n in c['contacts']])
    center=(pts.min(axis=0)+pts.max(axis=0))/2
    w,h=(750,630) if preview else (1800,1512)
    span=max(25.,(pts[:,1].max()-pts[:,1].min())*1.45,(pts[:,0].max()-pts[:,0].min())*h/w*1.4)
    view=set_view(center,span,w,h)
    cmd.png(str(OUT/f'{c["pdb"]}_local.png'),w,h,ray=1)
    c['local_view']=view
    c['rotation']=r.tolist()
    for s in c['sites']:
        s['local_point']=project(s['label_xyz'],r,view)
    for n in c['contacts']:
        n['local_point']=project(n['partner_xyz'],r,view)
    cmd.save(str(OUT/f'{c["pdb"]}_case.pse'))
    print(c['pdb'],'WT render and contact verification complete',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preview',action='store_true')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for c in CASES:
        render(c,args.preview)
    (OUT/'structure_audit.json').write_text(json.dumps(CASES,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'wt_contact_distances.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['PDB','mutated_WT_site','opposite_chain_residue','site_atom','opposite_atom','distance_A'])
        for c in CASES:
            for n in c['contacts']:
                writer.writerow([c['pdb'],n['site'],n['partner'],n['site_atom'],n['partner_atom'],f'{n["distance_A"]:.6f}'])


if __name__=='__main__':
    main()
