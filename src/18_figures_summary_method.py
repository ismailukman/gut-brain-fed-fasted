"""
Figure 13 (follow-up B): how the network value is formed changes the fed - fasted test (run after 14_followups.py).
  A  fed - fasted test statistic (estimate / standard error) for the region sets, their average and the Rebollo 2018
     regions, for three summaries: synchrony of the mean time course, mean of voxel synchrony, prominence
  B  fed - fasted estimates for CONN networks, region sets and the Rebollo regions, with network values computed as
     the mean of voxel synchrony

Usage: python 18_figures_summary_method.py
"""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gutsync import REGION_LABELS, store_caption

from config import BASE
RES = os.path.join(BASE, 'results')
SEND = os.environ.get('GUTBFF_SEND') == '1'
OUT = os.path.join(BASE, 'figures', 'send' if SEND else 'v2')
INK, SUB, GRY, GRID = '#1f1f1e', '#55554f', '#7a7a7a', '#ebeae6'
COL = {'PLV': '#6baed6', 'awPLV': '#08306b'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12.5, 'xtick.labelsize': 11, 'ytick.labelsize': 11.5,
                     'axes.edgecolor': GRY})
STYLE = {'Mean time course': ('#bdbdbd', 'o'), 'Mean of voxels': ('#6baed6', 's'), 'Prominence': ('#08306b', 'D')}
ORDER = ['average5', 'insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical', 'rebollo2018']
if SEND:
    ORDER = ORDER[:-1]
LAB = {**REGION_LABELS, 'average5': 'Average of the five sets', 'rebollo2018': 'Rebollo 2018 regions (prior map)'}
CONN_NAMES = {'DefaultMode': 'Default mode', 'SensoriMotor': 'Sensorimotor', 'Visual': 'Visual', 'Salience': 'Salience',
              'DorsalAttention': 'Dorsal attention', 'FrontoParietal': 'Frontoparietal', 'Language': 'Language',
              'Cerebellar': 'Cerebellar'}

st = pd.read_csv(os.path.join(RES, 'region_summaries_state.csv'))
st['z'] = st.fed_minus_fasted / st.se
fig = plt.figure(figsize=(14.5, 13.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.32, wspace=0.08, left=0.2, right=0.97, top=0.9, bottom=0.05)

# A: test statistic per summary
ys = -np.arange(len(ORDER), dtype=float)
ys[0] += 0.4
for j, metric in enumerate(('PLV', 'awPLV')):
    ax = fig.add_subplot(gs[0, j])
    for k, (summ, (c, mk)) in enumerate(STYLE.items()):
        d = st[(st.metric == metric) & (st.summary == summ)].set_index('region').loc[ORDER]
        sigq = d.q_fdr.fillna(1) < 0.05
        ax.scatter(d.z, ys + (k - 1) * 0.22, color=c, marker=mk, s=70, zorder=3, label=summ,
                   edgecolor=np.where(sigq, INK, c), linewidth=np.where(sigq, 1.8, 0.5))
    ax.axvspan(-1.96, 1.96, color='#f4f4f2', zorder=0)
    ax.axvline(0, color=GRY, lw=1)
    ax.set_xlim(-2.5, 4.2)
    ax.set_title(metric, fontsize=14, fontweight='bold', color=COL[metric], pad=10)
    ax.set_yticks(ys, [LAB[r] for r in ORDER] if j == 0 else [])
    ax.set_xlabel('Fed − fasted, estimate / standard error')
    ax.grid(axis='x', color=GRID)
    ax.spines[['top', 'right']].set_visible(False)
    if j == 1:
        fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc='upper right', bbox_to_anchor=(0.97, 0.955),
                   ncol=3, fontsize=11.5)
fig.text(0.01, 0.955, 'A', fontsize=17, fontweight='bold', color=INK)
fig.text(0.035, 0.955, 'Three ways of summarising a region set', fontsize=14, fontweight='bold', color=INK)
fig.text(0.035, 0.935, 'shaded: two-sided p > 0.05 · black edge: FDR q < 0.05 across the five sets', fontsize=11, color=SUB)

# B: mean of voxel synchrony per network
vs = pd.read_csv(os.path.join(RES, 'voxelmean_networks_state.csv'))
vs = vs[vs.scheme != 'Prior map'] if SEND else vs
nets = vs.drop_duplicates(['scheme', 'network'])[['scheme', 'network']].reset_index(drop=True)
nets['lab'] = [CONN_NAMES.get(n, n) if s == 'CONN' else LAB.get(n, n) for s, n in zip(nets.scheme, nets.network)]
gap = nets.scheme.map({'CONN': 0, 'Region set': 0.8, 'Prior map': 1.6}).values
yb = -(np.arange(len(nets)) + gap)
ax = fig.add_subplot(gs[1, :])
for mlab, off in (('PLV', 0.17), ('awPLV', -0.17)):
    d = vs[vs.metric == mlab].set_index(['scheme', 'network']).loc[list(zip(nets.scheme, nets.network))]
    ax.errorbar(d.fed_minus_fasted, yb + off, xerr=[d.fed_minus_fasted - d.ci_low, d.ci_high - d.fed_minus_fasted],
                fmt='o', color=COL[mlab], ms=7, elinewidth=2.2, capsize=0, label=mlab)
    for y, r in zip(yb + off, d.itertuples()):
        if pd.notna(r.q_fdr) and r.q_fdr < 0.05:
            ax.text(r.ci_high + 0.0015, y, f'q = {r.q_fdr:.3f}', va='center', fontsize=11, color=COL[mlab], fontweight='bold')
ax.axvline(0, color=GRY, lw=1)
ax.set_yticks(yb, nets.lab)
ax.set_xlabel('Fed − fasted, mean of voxel synchrony minus null mean (95% CI)')
ax.grid(axis='x', color=GRID)
ax.spines[['top', 'right']].set_visible(False)
for name, lab in [x for x in (('CONN', 'CONN networks'), ('Region set', 'Region sets'), ('Prior map', 'Prior map')) if (nets.scheme == x[0]).any()]:
    ax.text(-0.005, yb[nets.scheme == name].max() + 0.55, lab, transform=ax.get_yaxis_transform(), fontsize=11.5,
            color=SUB, style='italic', ha='right')
ax.legend(frameon=False, loc='lower right', fontsize=12)
fig.text(0.01, 0.505, 'B', fontsize=17, fontweight='bold', color=INK)
fig.text(0.035, 0.505, 'Network values as the mean of voxel synchrony', fontsize=14, fontweight='bold', color=INK)
fig.suptitle('How the network value is formed', fontsize=16, fontweight='bold', color=INK, x=0.01, ha='left', y=0.995)

name, label = 'fig13_summary_method', 'Figure 13'
caption = ('Effect of the summary method on the fed minus fasted comparison (follow-up B; 15 participants; mixed models with '
           'run, session order and mean framewise displacement). Synchrony is expressed as empirical minus the null mean. '
           '(A) Test statistic (estimate divided by its standard error) for synchrony of the region mean time course (grey '
           'circles), the mean of voxel synchrony (light blue squares) and prominence, the fraction of voxels above their '
           'own null 95th percentile (deep blue diamonds), for PLV and amplitude-weighted PLV (awPLV). (B) Fed minus fasted '
           'estimates with 95% confidence intervals for network values computed as the mean of voxel synchrony (PLV light '
           'blue, awPLV deep blue); q is shown where the FDR-corrected value within scheme and metric is below 0.05.'
           + ('' if SEND else ' The Rebollo et al. (2018) regions are a prior map and are not FDR-corrected.'))
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'{name}_nocap.{ext}'), dpi=250, facecolor='white', bbox_inches='tight', pad_inches=0.25)
store_caption(OUT, name, label, caption)
body = textwrap.fill(f'{label}: {caption}', width=140).replace(f'{label}:', rf'$\bf{{{label.replace(" ", "~")}:}}$', 1)
fig.text(0.5, 0.0, body, ha='center', va='top', fontsize=13, color=INK, linespacing=1.4)
fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=250, facecolor='white', bbox_inches='tight', pad_inches=0.25)
print('figure 13 written')
