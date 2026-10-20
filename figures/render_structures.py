"""Coordinate-based publication renders; no inferred mutant geometry."""
from pathlib import Path
import json
import copy
import shutil
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'runs' / 'figures'
ROOT.mkdir(parents=True, exist_ok=True)
STRUCTURES = REPO / 'data' / 'cases' / 'structures'

from figures.geometry_core import audit, frame, set_view, project, sel, CASES
from pymol import cmd

OUT = ROOT / 'render'
OUT.mkdir(exist_ok=True)
COLORS = {'heavy': '#456b91', 'light': '#3c949b', 'antigen': '#b7bdc4',
          'mutation': '#c95447', 'neighbor': '#c5a25f',
          'oxygen': '#d13a35', 'nitrogen': '#355dcc', 'sulfur': '#dfbd34'}


def rgb(h):
    return [int(h[i:i+2], 16)/255 for i in (1, 3, 5)]


def configure():
    for name, color in COLORS.items():
        cmd.set_color(name, rgb(color))
    cmd.bg_color('white')
    settings = {'orthoscopic': 1, 'field_of_view': 20,
                'ray_opaque_background': 1, 'antialias': 2,
                'ambient': .42, 'direct': .65, 'reflect': .10,
                'specular': .18, 'shininess': 40, 'ray_shadows': 0,
                'depth_cue': 0, 'cartoon_fancy_helices': 1,
                'cartoon_fancy_sheets': 1, 'cartoon_smooth_loops': 1,
                'cartoon_loop_radius': .18, 'cartoon_rect_length': 1.25,
                'cartoon_rect_width': .23, 'stick_radius': .16,
                'sphere_scale': .19, 'ray_trace_mode': 0,
                'stick_quality': 24, 'sphere_quality': 3}
    for name, value in settings.items():
        cmd.set(name, value)


def hetero_atoms(selection):
    for elem, color in [('N', 'nitrogen'), ('O', 'oxygen'), ('S', 'sulfur')]:
        cmd.color(color, f'({selection}) and elem {elem}')


def render(c):
    atoms, groups = audit(c)
    r = frame(c, groups)
    sitekeys = [(s['chain'], s['resi'], '') for s in c['sites']]
    # The three nearest inter-chain pairs for a, and the mutated pairs for b/c.
    if c['pdb'] == '1N8Z':
        drawn = [next(n for n in c['contacts'] if n['site'] == s['mutation'])
                 for s in c['sites']]
    else:
        drawn = [next(n for n in c['contacts'] if tuple(n['partner_key']) in sitekeys)]
    c['drawn_contacts'] = drawn
    c['rotation'] = r.tolist()
    cmd.reinitialize()
    cmd.load(str(STRUCTURES / (c['pdb'] + '.pdb')), 'complex')
    cmd.remove('not polymer.protein or hydro')
    cmd.remove('not alt "" and not alt A')
    cmd.alter('all', 'alt=""')
    # A rigid right-handed rotation only; coordinates are otherwise unmodified.
    cmd.alter_state(1, 'all', '(x,y,z)=tuple(r @ np.array([x,y,z]))',
                    space={'r': r, 'np': np})
    configure()
    cmd.hide('everything')
    for chain, color in [('A', 'light'), ('B', 'heavy'), ('C', 'antigen')]:
        cmd.color(color, f'chain {chain}')
        cmd.set('cartoon_color', color, f'chain {chain}')
    cmd.select('mut_sites', ' or '.join(sel(k) for k in sitekeys))
    cmd.show('cartoon', 'all')
    cmd.show('sticks', 'mut_sites and (sidechain or name CA)')
    cmd.color('mutation', 'mut_sites')
    hetero_atoms('mut_sites')
    xyz = np.stack([r @ a['xyz'] for a in atoms if a['name'] == 'CA'])
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    overview_center = (lo + hi) / 2
    span = max(hi[1]-lo[1], hi[0]-lo[0]) * 1.15
    view = set_view(overview_center, span, 1000, 1000)
    cmd.png(str(OUT / (c['pdb'] + '_overview.png')), 1000, 1000, ray=1)
    c['overview_view'] = view
    for s in c['sites']:
        s['overview_point'] = project(s['label_xyz'], r, view)

    cmd.hide('cartoon', 'all')
    cmd.select('local', 'byres (complex within 10.0 of mut_sites)')
    cmd.show('cartoon', 'local')
    cmd.set('cartoon_transparency', .37, 'local')
    cmd.set('cartoon_transparency', .47, 'chain C')
    neighbor_keys = {tuple(n['partner_key']) for n in drawn} - set(sitekeys)
    if neighbor_keys:
        cmd.select('neighbors', ' or '.join(sel(k) for k in neighbor_keys))
        cmd.color('neighbor', 'neighbors')
        # Include backbone atoms, so the measured P571 carbonyl O is visible.
        cmd.show('sticks', 'neighbors')
        cmd.show('spheres', 'neighbors')
        hetero_atoms('neighbors')
    cmd.color('mutation', 'mut_sites')
    cmd.show('sticks', 'mut_sites and (sidechain or name CA)')
    cmd.show('spheres', 'mut_sites and (sidechain or name CA)')
    hetero_atoms('mut_sites')
    for i, n in enumerate(drawn):
        a = sel(n['site_key']) + f' and name {n["site_atom"]}'
        b = sel(n['partner_key']) + f' and name {n["partner_atom"]}'
        obj = f'contact_{i+1}'
        distance = cmd.distance(obj, a, b, mode=0)
        if abs(distance - n['distance_A']) > .002:
            raise ValueError('Distance changed during rotation')
        cmd.hide('labels', obj)
        for setting, value in [('dash_color', 'gray40'), ('dash_gap', .22),
                               ('dash_length', .18), ('dash_width', 1.7)]:
            cmd.set(setting, value, obj)
    points = np.stack([r @ a['xyz'] for k in sitekeys for a in groups[k]] +
                      [r @ np.asarray(n['partner_xyz']) for n in drawn])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    view = set_view(center, 24.0, 1800, 1500)
    cmd.png(str(OUT / (c['pdb'] + '_local.png')), 1800, 1500, ray=1)
    c['local_view'] = view
    for s in c['sites']:
        s['local_point'] = project(s['label_xyz'], r, view)
    c['partner_labels'] = []
    for k in sorted(neighbor_keys):
        residue = groups[k]
        side = [a['xyz'] for a in residue if a['name'] not in ('N', 'CA', 'C', 'O')]
        centroid = np.mean(side if side else [a['xyz'] for a in residue], axis=0)
        n = next(n for n in drawn if tuple(n['partner_key']) == k)
        c['partner_labels'].append({'label': n['partner'],
                                    'local_point': project(centroid, r, view)})
    for n in drawn:
        n['site_point'] = project(n['site_xyz'], r, view)
        n['partner_point'] = project(n['partner_xyz'], r, view)
    cmd.save(str(OUT / (c['pdb'] + '_publication.pse')))
    print(c['pdb'], 'coordinate render complete', flush=True)


def main():
    cases = copy.deepcopy(CASES)
    for c in cases:
        render(c)
    (OUT / 'structure_audit.json').write_text(json.dumps(cases, indent=2) + '\n')
    (ROOT / 'render_palette.json').write_text(json.dumps(COLORS, indent=2) + '\n')


if __name__ == '__main__':
    main()
