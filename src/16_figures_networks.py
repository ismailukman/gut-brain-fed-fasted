"""
Figures for the network-level EGG-BOLD synchrony analysis (run after 11_network_synchrony.py).

fig6_network_conn      empirical vs null PLV/awPLV densities, 8 CONN ICA-derived networks
fig7_network_regions   the same for the five region sets and the Rebollo 2018 gastric-coupled regions
fig8_network_state     fed minus fasted estimates per network (both schemes, PLV and awPLV)

Usage: python 16_figures_networks.py
"""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

from gutsync import REGION_LABELS, store_caption

from config import BASE
NET = os.path.join(BASE, 'results', 'networks')
SEND = os.environ.get('GUTBFF_SEND') == '1'
OUT = os.path.join(BASE, 'figures', 'send' if SEND else 'v2')
INK, SUB, GRY, GRID = '#1f1f1e', '#55554f', '#7a7a7a', '#ebeae6'
COL = {'plv': '#6baed6', 'awplv': '#08306b'}   # PLV light blue, awPLV deep blue
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12.5, 'axes.labelsize': 12.5,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.titlesize': 13, 'axes.edgecolor': GRY})

runs = pd.read_csv(os.path.join(NET, 'network_runs.csv'))
pop = pd.read_csv(os.path.join(NET, 'population_level_networks.csv'))
sta = pd.read_csv(os.path.join(NET, 'fed_vs_fasted_networks.csv'))
pool = np.load(os.path.join(NET, 'null_pool.npy'), allow_pickle=True).item()
CONN_NAMES = {'DefaultMode': 'Default mode', 'SensoriMotor': 'Sensorimotor', 'Visual': 'Visual',
              'Salience': 'Salience', 'DorsalAttention': 'Dorsal attention', 'FrontoParietal': 'Frontoparietal',
              'Language': 'Language', 'Cerebellar': 'Cerebellar'}


def label(scheme, net):
    return CONN_NAMES.get(net, net) if scheme == 'CONN' else REGION_LABELS.get(net, net)


def stars(q):
    return '' if pd.isna(q) else '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else ''


def save(fig, name, lab, caption):
    for ext in ('png', 'pdf'):               # copy without the caption, for use in documents
        fig.savefig(os.path.join(OUT, f'{name}_nocap.{ext}'), dpi=250, facecolor='white', bbox_inches='tight', pad_inches=0.25)
    store_caption(OUT, name, lab, caption)
    width = int(fig.get_size_inches()[0] * 7.6)
    body = textwrap.fill(f'{lab}: {caption}', width=width)
    body = body.replace(f'{lab}:', rf'$\bf{{{lab.replace(" ", "~")}:}}$', 1)
    fig.text(0.5, -0.01, body, ha='center', va='top', fontsize=13, color=INK, linespacing=1.4)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=250, facecolor='white', bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


def density_grid(scheme, nets, name, lab, title, extra):
    ncol_pairs = 2
    nrow = int(np.ceil(len(nets) / ncol_pairs))
    fig = plt.figure(figsize=(12.5, 3.3 * nrow + 0.8))
    outer = fig.add_gridspec(nrow, ncol_pairs, wspace=0.22, hspace=0.7, left=0.07, right=0.98, top=1 - 1.3 / (3.3 * nrow + 0.8), bottom=0.05)
    for i, net in enumerate(nets):
        inner = outer[i // ncol_pairs, i % ncol_pairs].subgridspec(1, 2, wspace=0.1)
        g = runs[(runs.scheme == scheme) & (runs.network == net)]
        for k, metric in enumerate(('plv', 'awplv')):
            ax = fig.add_subplot(inner[k])
            emp = g[metric].dropna().values
            nul = np.asarray(pool[f'{scheme}|{net}'][metric], dtype=float)
            nul = nul[np.isfinite(nul)]
            hi = max(np.percentile(nul, 99.7), emp.max()) * 1.08
            xs = np.linspace(0, hi, 300)
            ax.fill_between(xs, stats.gaussian_kde(emp)(xs), color=COL[metric], alpha=0.45, lw=0)
            ax.plot(xs, stats.gaussian_kde(emp)(xs), color=COL[metric], lw=2, label='Empirical')
            ax.plot(xs, stats.gaussian_kde(nul)(xs), color=GRY, lw=1.8, ls='--', label='Null')
            r = pop[(pop.scheme == scheme) & (pop.network == net) & (pop.metric == ('PLV' if metric == 'plv' else 'awPLV'))].iloc[0]
            q = r.q_fdr if pd.notna(r.q_fdr) else r.p_value
            qlab = 'q' if pd.notna(r.q_fdr) else 'p'
            qtxt = f'{qlab} < 0.001' if q < 0.001 else f'{qlab} = {q:.3f}'
            ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
            ax.text(0.98, 0.98, f'Δ {r.mean_empirical_minus_null:+.3f}'.replace('-', '−') + '\n' + f'{qtxt} {stars(r.q_fdr)}'.strip(),
                    transform=ax.transAxes, ha='right', va='top', fontsize=11.5, color=INK, linespacing=1.3)
            ax.set_xlabel('PLV' if metric == 'plv' else 'awPLV')
            ax.set_yticks([])
            ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.text(inner[0].get_position(fig).x0, inner[0].get_position(fig).y1 + 0.012,
                 label(scheme, net), fontsize=14, fontweight='bold', color=INK)
    summary = len(nets) % ncol_pairs == 1          # an empty slot: use it for a summary of all networks
    if summary:
        ax = fig.add_subplot(outer[nrow - 1, ncol_pairs - 1])
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])
        ys = np.arange(len(nets))[::-1]
        for metric, off in (('plv', 0.15), ('awplv', -0.15)):
            for y, net in zip(ys, nets):
                g = runs[(runs.scheme == scheme) & (runs.network == net)]
                d = g.groupby('subject_id')[f'd{metric}'].mean().dropna()
                m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
                h = stats.t.ppf(0.975, len(d) - 1) * se
                ax.plot([m - h, m + h], [y + off] * 2, color=COL[metric], lw=2.4, solid_capstyle='round')
                ax.plot(m, y + off, 'o', color=COL[metric], ms=7, mec='white', mew=1.2)
        ax.axvline(0, color=GRY, lw=1)
        ax.set_yticks(ys, [label(scheme, n).replace(' network', '').replace(' nuclei', '').replace('-', ' ') for n in nets])
        ax.set_xlabel('Δ, empirical − null mean (95% CI)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color=GRID)
        fig.text(ax.get_position(fig).x0, ax.get_position(fig).y1 + 0.012, 'Summary: synchrony above chance',
                 fontsize=14, fontweight='bold', color=INK)
    hs = [plt.Rectangle((0, 0), 1, 1, color=COL['plv'], alpha=0.6), plt.Rectangle((0, 0), 1, 1, color=COL['awplv'], alpha=0.6),
          plt.Line2D([], [], color=GRY, ls='--', lw=1.8)]
    fig.legend(hs, ['Empirical PLV', 'Empirical awPLV', 'Null (other participants\' EGG)'], frameon=False, ncol=3,
               fontsize=12, loc='upper left', bbox_to_anchor=(0.012, 1.0 - 0.32 / fig.get_size_inches()[1]))
    fig.suptitle(title, fontsize=16, fontweight='bold', color=INK, x=0.02, ha='left', y=0.995)
    save(fig, name, lab,
         'Gaussian kernel density estimates (smoothed histograms) of synchronization between the gastric rhythm and '
         f'{extra} (all usable runs, 20 participants, both states). Empirical distributions are filled (PLV light blue, awPLV deep blue); '
         'distributions obtained by pairing each run with other participants\' EGG (null) are grey and dashed. Each '
         'network value is the mean over its regions of interest. Δ: participant mean of empirical minus null mean '
         '(synchrony above chance). q: one-sided Wilcoxon signed-rank test of Δ against zero, FDR-corrected across '
         'networks; * q < 0.05, ** q < 0.01, *** q < 0.001.'
         + (' The last panel summarises Δ for all region sets with 95% confidence intervals across participants '
            '(PLV light blue, awPLV deep blue).' if len(nets) % 2 == 1 else ''))


conn_nets = [n for n in CONN_NAMES if n in set(runs[runs.scheme == 'CONN'].network)]
reg_nets = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical'] + ([] if SEND else ['rebollo2018'])
density_grid('CONN', conn_nets, 'fig6_network_conn', 'Figure 6', 'Gastric synchrony of CONN ICA-derived networks',
             'eight CONN ICA-derived resting-state networks')
density_grid('Region', reg_nets, 'fig7_network_regions', 'Figure 7', 'Gastric synchrony of the predefined region sets',
             ('the five predefined region sets' if SEND else 'the five predefined region sets and the regions coupled to the gastric rhythm in Rebollo et al. (2018), which are shown with an uncorrected p because it was a separate a priori test'))

# fed minus fasted per network
fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=False)
for ax, scheme, nets in ((axes[0], 'CONN', conn_nets), (axes[1], 'Region', reg_nets)):
    ys = np.arange(len(nets))[::-1]
    for j, (metric, col, off) in enumerate((('PLV', COL['plv'], 0.16), ('awPLV', COL['awplv'], -0.16))):
        d = sta[(sta.scheme == scheme) & (sta.metric == metric)].set_index('network').loc[nets]
        ax.errorbar(d.fed_minus_fasted, ys + off, xerr=[d.fed_minus_fasted - d.ci_low, d.ci_high - d.fed_minus_fasted],
                    fmt='o', color=col, ms=7, elinewidth=2, capsize=0, label=metric)
        for y, (_, r) in zip(ys + off, d.iterrows()):
            if pd.notna(r.q_fdr) and r.q_fdr < 0.05:
                ax.text(r.ci_high + 0.004, y, stars(r.q_fdr), va='center', color=col, fontsize=12)
    ax.axvline(0, color=GRY, lw=1)
    ax.set_yticks(ys, [label(scheme, n) for n in nets])
    ax.set_xlabel('Fed − fasted (empirical − null), 95% CI')
    ax.set_title('CONN ICA-derived networks' if scheme == 'CONN' else 'Region sets', fontsize=14, fontweight='bold')
    ax.grid(axis='x', color=GRID)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
fig.suptitle('Effect of feeding on network synchrony', fontsize=16, fontweight='bold', x=0.02, ha='left')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc='upper right', ncol=2, fontsize=13, bbox_to_anchor=(0.98, 0.985))
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, 'fig8_network_state', 'Figure 8',
     'Fed minus fasted difference in network-level synchrony (empirical minus null mean) in the 15 participants with '
     'usable EGG in both states, for PLV (light blue) and awPLV (deep blue), with 95% confidence intervals from linear mixed '
     'models (state, run, session order, mean framewise displacement, random intercept per participant). Left: '
     'CONN ICA-derived networks; right: predefined region sets' + ('. ' if SEND else ' and the Rebollo et al. (2018) gastric-coupled regions. ')
     + (f'No difference survived FDR correction across networks (smallest q = {sta.q_fdr.min():.3f}).'
        if sta.q_fdr.min() >= 0.05 else 'Asterisks mark FDR-corrected q < 0.05 across networks.'))
print('network figures written')
