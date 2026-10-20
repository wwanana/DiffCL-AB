"""183-mm publication artwork with editable text, true-coordinate images and data."""
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
from PIL import Image
import fitz

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'runs' / 'figures'
DATA = REPO / 'data' / 'cases'
(ROOT/'data').mkdir(parents=True,exist_ok=True)
font='DejaVu Sans'
font_root=Path('/usr/share/fonts/opentype/urw-base35')
if (font_root/'NimbusSans-Regular.otf').exists():
    for name in ('NimbusSans-Regular.otf','NimbusSans-Bold.otf','NimbusSans-Italic.otf'):
        font_manager.fontManager.addfont(str(font_root/name))
    font='Nimbus Sans'
plt.rcParams.update({'font.family': font, 'font.size': 6.5,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.linewidth': .55, 'axes.labelsize': 6.5,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
                     'xtick.major.width': .55, 'ytick.major.width': .55,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'text.color': '#1d242a', 'axes.labelcolor': '#1d242a',
                     'xtick.color': '#1d242a', 'ytick.color': '#1d242a',
                     'mathtext.default': 'regular'})
W = 183.0
INK = '#1d242a'
TEAL = '#258991'
MUT = '#c95447'
X = (4., 64., 124.)
COL = 55.


def txt(fig, H, x, y, s, size=6.5, weight='normal', ha='left', **kwargs):
    return fig.text(x/W, y/H, s, fontsize=size, color=INK, weight=weight,
                    ha=ha, va='baseline', **kwargs)


def rule(fig, H, y, x1=4, x2=179, color='#929ba1', lw=.45):
    fig.add_artist(plt.Line2D([x1/W, x2/W], [y/H, y/H],
                             transform=fig.transFigure, color=color, lw=lw))


def metrics(frame):
    a, b = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    return {'n': len(a), 'pcc': float(pearsonr(a, b).statistic),
            'spearman': float(spearmanr(a, b).statistic),
            'rmse': float(np.sqrt(np.mean((a-b)**2)))}


def residue_labels(ax, case):
    # Short labels and black keylines conform to scientific figure conventions.
    positions = {
        '1N8Z': {'A:N30': (.09, .24), 'A:H91': (.08, .78),
                 'A:Y92': (.51, .91), 'C:D596': (.91, .17),
                 'C:P571': (.87, .41), 'C:K569': (.91, .63)},
        '1DQJ': {'B:W98': (.13, .70), 'C:Y20': (.89, .37)},
        '1VFB': {'A:Y32': (.12, .71), 'C:Q121': (.88, .48)}
    }
    labels = [{'label': f'{s["chain"]}:{s["wt"]}{s["resi"]}',
               'local_point': s['local_point']} for s in case['sites']]
    labels += case['partner_labels']
    for l in labels:
        pos = positions[case['pdb']][l['label']]
        ax.annotate(l['label'], xy=l['local_point'], xycoords='data',
                    xytext=pos, textcoords='axes fraction', color=INK,
                    ha='center', va='center', fontsize=6.2,
                    bbox={'boxstyle': 'square,pad=.13', 'fc': 'white',
                          'ec': 'none', 'alpha': .94},
                    arrowprops={'arrowstyle': '-', 'color': '#495259',
                                'lw': .45, 'shrinkA': 2, 'shrinkB': 2})


def top_row(fig, H, cases, selected, palette):
    ly = H - 4.8
    for x, key, label in [(4, 'heavy', 'Heavy (B)'), (35, 'light', 'Light (A)'),
                          (65, 'antigen', 'Antigen (C)'),
                          (100, 'mutation', 'WT mutation site'),
                          (145, 'neighbor', 'Contact residue')]:
        fig.add_artist(plt.Line2D([(x+.1)/W, (x+2.7)/W],
                                 [(ly+.6)/H, (ly+.6)/H],
                                 transform=fig.transFigure, color=palette[key],
                                 lw=2.7, solid_capstyle='butt'))
        txt(fig, H, x+4.1, ly, label, 6.3)
    for i, (x, c, row) in enumerate(zip(X, cases, selected.itertuples(index=False))):
        txt(fig, H, x, H-12.4, chr(97+i), 8, 'bold')
        txt(fig, H, x+4.2, H-12.4, c['title'].replace('–', '-'), 7, 'bold')
        txt(fig, H, x+COL, H-12.4, c['pdb'], 6.2, ha='right')
        txt(fig, H, x+4.2, H-16.7, row.mutation.replace(',', '  '), 6.2)
        # A fixed physical size preserves readability in both artwork versions.
        image_y = H - 64.
        ax = fig.add_axes([x/W, image_y/H, COL/W, (COL*1500/1800)/H])
        ax.imshow(Image.open(ROOT / 'render' / (c['pdb'] + '_local.png')))
        ax.axis('off')
        residue_labels(ax, c)
        inset_size = 11.0
        inset = fig.add_axes([(x+COL-inset_size)/W, (H-30.6)/H,
                              inset_size/W, inset_size/H])
        inset.imshow(Image.open(ROOT / 'render' / (c['pdb'] + '_overview.png')))
        pts = np.asarray([s['overview_point'] for s in c['sites']])
        p = pts.mean(axis=0)
        radius = max(35., float(np.linalg.norm(pts-p, axis=1).max()) + 20.)
        inset.add_patch(Circle(p, radius, ec=INK, fc='none', lw=.45))
        inset.axis('off')
        contacts = c['drawn_contacts']
        if c['pdb'] == '1N8Z':
            lines = ['N30-D596 2.99 Å   H91-P571 2.89 Å', 'Y92-K569 2.96 Å']
        else:
            n = contacts[0]
            wt = n['site'].split(':')[1][:-1]
            partner = n['partner'].split(':')[1]
            lines = [f'{wt}-{partner} {n["distance_A"]:.2f} Å']
        for j, line in enumerate(lines):
            txt(fig, H, x+COL/2, H-66.7-j*3.0, line, 5.8, ha='center')
    # Compact common table; units stated once and no decorative cards.
    table_top = H - 73.4
    txt(fig, H, 4, table_top+1.1, 'ΔΔG (kcal/mol)', 6.2)
    for x, row in zip(X, selected.itertuples(index=False)):
        rule(fig, H, table_top, x, x+COL)
        cols = [x+9., x+27.5, x+46.]
        for xx, label, val in zip(cols, ['Experiment', 'DiffCL-AB', '|Error|'],
                                  [row.y_true, row.y_pred, row.abs_error]):
            txt(fig, H, xx, table_top-3.4, label, 6.1, ha='center')
            value = f'{val:+.2f}' if label != '|Error|' else f'{val:.2f}'
            txt(fig, H, xx, table_top-7.4, value, 7, 'bold', ha='center')
        rule(fig, H, table_top-10.1, x, x+COL, color='#c0c6ca', lw=.35)


def plots(fig, H, pred, selected):
    quant = pred[~np.isclose(pred.y_true, 8., atol=1e-6, rtol=0)]
    groups = [('All quantitative test', quant),
              ('Seen complexes', quant[quant.complex_context == 'seen']),
              ('Unseen complexes', quant[quant.complex_context == 'unseen'])]
    stat = {}
    for i, (x, (title, frame)) in enumerate(zip(X, groups)):
        txt(fig, H, x, 63.5, chr(100+i), 8, 'bold')
        txt(fig, H, x+4.2, 63.5, title, 7, 'bold')
        ax = fig.add_axes([(x+8)/W, 10.5/H, 44/W, 44/H])
        ax.scatter(frame.y_true, frame.y_pred, s=7.0, c=TEAL, alpha=.50,
                   edgecolors='white', linewidths=.18, zorder=2)
        ax.plot([-4, 9], [-4, 9], c='#757e85', lw=.65, ls=(0, (3, 3)), zorder=1)
        ax.set_xlim(-4, 9); ax.set_ylim(-4, 9)
        ax.set_xticks([-4, 0, 4, 8]); ax.set_yticks([-4, 0, 4, 8])
        ax.set_xlabel('Experimental ΔΔG (kcal/mol)', labelpad=2.8)
        ax.set_ylabel('Predicted ΔΔG (kcal/mol)', labelpad=3.0)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#4b555e')
        m = metrics(frame)
        stat[title] = m
        # Keep statistics outside the data region so no outlier is obscured.
        txt(fig, H, x+8, 60.1,
            f'n = {m["n"]}   r = {m["pcc"]:.3f}   ρ = {m["spearman"]:.3f}', 5.8)
        txt(fig, H, x+8, 57.4, f'RMSE = {m["rmse"]:.3f}', 5.8)
        if i == 1:
            for j, (row, marker) in enumerate(zip(selected.itertuples(index=False), ('o', 's', 'D'))):
                ax.scatter([row.y_true], [row.y_pred], marker=marker, s=24,
                           facecolors=MUT, edgecolors='white', lw=.55, zorder=4)
                offset = (4, 5) if j != 2 else (5, -7)
                ax.annotate(chr(97+j), (row.y_true, row.y_pred), xytext=offset,
                            textcoords='offset points', fontsize=6.2, fontweight='bold',
                            color=INK, zorder=5)
    return stat


def export(full, cases, selected, pred, palette):
    H = 151. if full else 90.
    name = 'DiffCL_AB_Publication_Full' if full else 'DiffCL_AB_Three_Cases'
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
    top_row(fig, H, cases, selected, palette)
    stat = plots(fig, H, pred, selected) if full else None
    fig.canvas.draw()
    # Detect out-of-canvas text at the final publication size.
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_text() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer)
        bounds = fig.bbox
        if box.x0 < bounds.x0-.5 or box.y0 < bounds.y0-.5 or box.x1 > bounds.x1+.5 or box.y1 > bounds.y1+.5:
            raise ValueError('Text extends outside artwork: ' + text.get_text())
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(ROOT / f'{name}.{ext}', dpi=600, facecolor='white', bbox_inches=None,
                    metadata={'Creator': 'Coordinate-based PyMOL and Matplotlib scientific figure'} if ext == 'pdf' else None)
    plt.close(fig)
    png = Image.open(ROOT / f'{name}.png').convert('RGB')
    png.save(ROOT / f'{name}.tiff', compression='tiff_lzw', dpi=(600, 600))
    pdf = fitz.open(ROOT / f'{name}.pdf')
    assert len(pdf) == 1
    page = pdf[0]
    assert abs(page.rect.width/72*25.4-W) < .02
    assert abs(page.rect.height/72*25.4-H) < .02
    page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(ROOT / 'render' / f'{name}_pdf_preview.png')
    if full:
        (ROOT / 'data/displayed_metrics.json').write_text(json.dumps(stat, indent=2)+'\n')


def main():
    cases = json.loads((ROOT / 'render/structure_audit.json').read_text())
    selected = pd.read_csv(DATA / 'case_predictions.csv')
    pred = pd.read_csv(REPO / 'results/archived/mutd_predictions.csv')
    palette = json.loads((ROOT / 'render_palette.json').read_text())
    assert list(selected.pdb_id) == ['1N8Z', '1DQJ', '1VFB']
    assert list(selected.source_index) == [104, 4, 363]
    assert len(pred) == 455
    assert np.isclose(pred.y_true, 8., atol=1e-6, rtol=0).sum() == 23
    assert np.allclose(selected.abs_error, abs(selected.y_pred-selected.y_true), atol=1e-12)
    for c, row in zip(cases, selected.itertuples(index=False)):
        assert ','.join(c['mutations']) == row.mutation
        assert row.complex_context == 'seen'
        matching = pred[pred.source_index == row.source_index].iloc[0]
        assert row.y_pred == matching.y_pred and row.y_true == matching.y_true
    export(True, cases, selected, pred, palette)
    export(False, cases, selected, pred, palette)
    provenance = {'case_rows': [104, 4, 363], 'test_n': len(pred),
                  'quantitative_n': 432, 'assigned_plus8_n': 23,
                  'prediction_values_modified': False,
                  'mutant_structures_generated': False,
                  'width_mm': W, 'full_height_mm': 151, 'case_height_mm': 90,
                  'png_dpi': 600, 'model_predictions': 'Existing formal test output',
                  'case_table_sha256': hashlib.sha256((DATA / 'case_predictions.csv').read_bytes()).hexdigest(),
                  'test_table_sha256': hashlib.sha256((REPO / 'results/archived/mutd_predictions.csv').read_bytes()).hexdigest()}
    (ROOT / 'data/figure_validation.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Two publication layouts exported; numerical and canvas checks passed.')


if __name__ == '__main__':
    main()
