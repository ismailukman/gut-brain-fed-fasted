"""
Figure 12 (follow-up A): whole-brain fed minus fasted maps for PLV and awPLV (run after 14_followups.py).
The paired t of every voxel is shown unthresholded on the inflated fsaverage5 surface (lateral and medial views),
with the largest cluster from the cluster-based permutation test (cluster-forming p < 0.01) outlined in black.

Usage: python 17_figures_cluster.py
"""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from gutsync import store_caption
from surface_render import surface_panel

from config import BASE
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, 'figures', 'v2')
MAPS = os.path.join(RES, 'synchrony', 'maps_v2')
INK, SUB = '#1f1f1e', '#55554f'
COL = {'PLV': '#6baed6', 'awPLV': '#08306b'}
TLIM = 4.0
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12.5})

reg = pd.read_csv(os.path.join(RES, 'synchrony', 'region_plv_awplv_runs.csv'))
paired = reg.drop_duplicates(['imaging_id', 'run'])
paired = paired[paired.in_primary_analysis]
cl = pd.read_csv(os.path.join(RES, 'cluster_second_level.csv'))


def tmap(m):
    load = lambda a, b: np.asanyarray(nib.load(os.path.join(MAPS, f'{a}_run{b}_{m}_dmean.nii.gz')).dataobj).astype(float)
    d = []
    for _, g in paired.groupby('subject_id'):
        st = {s: np.mean([load(a, b) for a, b in zip(gg.imaging_id, gg.run)], 0) for s, gg in g.groupby('state')}
        d.append(st['fed'] - st['fasted'])
    d = np.stack(d)
    sd = d.std(0, ddof=1)
    return np.where(sd > 0, d.mean(0) / (sd / np.sqrt(len(d)) + 1e-12), 0), len(d)


VIEWS = [('lh', 'lateral', 'Left lateral'), ('lh', 'medial', 'Left medial'),
         ('rh', 'medial', 'Right medial'), ('rh', 'lateral', 'Right lateral')]
fig = plt.figure(figsize=(16, 8.4))
for i, (m, mlab) in enumerate((('plv', 'PLV'), ('awplv', 'awPLV'))):
    lab = nib.load(os.path.join(RES, f'cluster_{m}_fed_minus_fasted_p0.01.nii.gz'))
    rows = cl[(cl.metric == mlab) & (cl.test == 'fed_minus_fasted') & (cl.p_cluster_forming == 0.01)].sort_values('p_fwe')
    b = rows.iloc[0]
    t, n = tmap(m)
    timg = nib.Nifti1Image(t.astype(np.float32), lab.affine)
    outline = nib.Nifti1Image((np.asanyarray(lab.dataobj) == b.cluster).astype(np.int16), lab.affine)
    y0 = 0.5 - i * 0.42
    for j, (h, vw, vl) in enumerate(VIEWS):
        ax = fig.add_axes([0.07 + j * 0.205, y0, 0.2, 0.38], projection='3d')
        norm, cm = surface_panel(ax, timg, h, vw, 'RdBu_r', -TLIM, TLIM, outline_img=outline)
        if i == 0:
            ax.set_title(vl, fontsize=12.5, color=SUB, pad=-4)
    fig.text(0.025, y0 + 0.19, mlab, fontsize=16, fontweight='bold', color=COL[mlab], rotation=90, va='center')
    fig.text(0.08, y0 + 0.005,
             f'Outline: largest cluster, {int(b.n_voxels)} voxels, peak MNI ({b.peak_mni}), '
             + (f'p$_{{FWE}}$ = {b.p_fwe:.4f}' if 0.0495 <= b.p_fwe < 0.0505 else f'p$_{{FWE}}$ = {b.p_fwe:.3f}') + (' (not significant)' if b.p_fwe >= 0.05 else ''),
             fontsize=11.5, color=SUB)
cax = fig.add_axes([0.915, 0.18, 0.012, 0.6])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, extend='both')
cb.set_label(f'Fed − fasted, paired t (df = {n - 1})', fontsize=12.5)
cb.ax.tick_params(labelsize=11)
fig.suptitle('Whole-brain effect of feeding on stomach-brain synchrony', fontsize=16, fontweight='bold', color=INK,
             x=0.02, ha='left', y=0.99)

name, label = 'fig12_cluster_maps', 'Figure 12'
caption = (f'Whole-brain fed minus fasted difference in stomach-brain synchrony (follow-up A; {n} participants). Each map '
           'shows the paired t statistic of every voxel for PLV (top) and amplitude-weighted PLV (awPLV, bottom), both '
           'expressed as empirical minus the null mean, without thresholding (red, higher after feeding; blue, lower). '
           'The black outline marks the largest cluster from the cluster-based sign-flip permutation test (cluster-forming '
           'p < 0.01, 5000 permutations); no cluster reached family-wise error p < 0.05. Inflated fsaverage surface; '
           'values were sampled between the white and pial surfaces.')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'{name}_nocap.{ext}'), dpi=250, facecolor='white', bbox_inches='tight', pad_inches=0.25)
store_caption(OUT, name, label, caption)
body = textwrap.fill(f'{label}: {caption}', width=160).replace(f'{label}:', rf'$\bf{{{label.replace(" ", "~")}:}}$', 1)
fig.text(0.5, -0.01, body, ha='center', va='top', fontsize=13, color=INK, linespacing=1.4)
fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=250, facecolor='white', bbox_inches='tight', pad_inches=0.25)
print('figure 12 written')
