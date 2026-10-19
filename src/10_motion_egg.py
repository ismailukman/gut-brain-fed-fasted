"""
EGG-head motion phase synchrony.

Per run and motion parameter: PLV and awPLV between the EGG and the motion trace, both band-passed at the
EGG's filter frequency +/- 0.015 Hz, against a null of the same motion trace with every passing EGG run of the
other participants.
  run level    p = (1 + n_null >= empirical) / (1 + n_null), never 0
  population   one value per participant (mean over runs of empirical minus null mean), one-sided Wilcoxon
               signed-rank (above chance); FDR and Bonferroni across the six parameters, separately per metric
  state        fed vs fasted mixed model on the paired set (run, session order, random intercept)
  by state     the population test repeated within each state (fasted only, fed only), with the null restricted
               to other participants' EGG from the same state
AFNI 3dvolreg column order: roll, pitch, yaw (deg), dS, dL, dP (mm).

Outputs: results/aim2_motion_egg_control.csv (population and state tests), results/synchrony/motion_egg_runs.csv,
         results/motion/population_level_gutbff.csv and motion_summary_gutbff.csv,
         results/motion/population_level_gutbff_{fasted,fed}.csv, results/aim2_motion_egg_by_state.csv,
         figures/v2/fig9_motion_egg.png (both states), fig10_motion_egg_fasted.png, fig11_motion_egg_fed.png
Usage: python 10_motion_egg.py
"""
import os
import textwrap
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from gutsync import analytic, bandpass, egg_peak, load_egg, store_caption, fit_mixed

warnings.filterwarnings('ignore')
from config import BASE
from config import RAW as ROOT
PARAMS = ['roll', 'pitch', 'yaw', 'dS (superior)', 'dL (left)', 'dP (posterior)']

man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
man = man[man.egg_pass].reset_index(drop=True)
info = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_runs.csv'))
man = man.merge(info[['imaging_id', 'run', 'mean_fd_mm', 'session_order', 'motion_file']], on=['imaging_id', 'run'])


def unit(z):
    return z / np.abs(z)


def plv_aw(zm, zg):
    """zm: T x P motion analytic, zg: T x K EGG analytic -> PLV and awPLV, each P x K."""
    um, ug = unit(zm), unit(zg)
    plv = np.abs(um.T @ np.conj(ug)) / len(zm)
    w = np.abs(zm)[:, :, None] * np.abs(zg)[:, None, :]                      # T x P x K
    aw = np.abs(np.sum(w * um[:, :, None] * np.conj(ug)[:, None, :], 0)) / w.sum(0)
    return plv, aw


egg = {(r.imaging_id, r.run): load_egg(os.path.join(ROOT, r.egg_filtered_file)) for r in man.itertuples()}
STATE = {(r.imaging_id, r.run): r.state for r in man.itertuples()}
null_pool = {prm: {'plv': [], 'awplv': []} for prm in PARAMS}
null_state = {(st, prm): {'plv': [], 'awplv': []} for st in ('fasted', 'fed') for prm in PARAMS}
rows, summary = [], []
for r in man.itertuples():
    f0 = egg_peak(os.path.join(ROOT, r.egg_filtered_file))
    g = egg[(r.imaging_id, r.run)]
    n = len(g)
    mot = np.loadtxt(os.path.join(ROOT, r.motion_file))[:n]
    mot_bp = bandpass(mot - mot.mean(0), f0, 0.5)
    zm = analytic(mot_bp)
    summary.append({'imaging_id': r.imaging_id, 'run': r.run, 'state': r.state,
                    **dict(zip(PARAMS, np.abs(mot_bp).mean(0)))})
    zg = analytic(bandpass(g, f0, 0.5))[:, None]
    keys = [k for k in egg if k[0][:9] != r.imaging_id[:9] and len(egg[k]) >= n]
    zp = np.column_stack([analytic(bandpass(egg[k][:n], f0, 0.5)) for k in keys])
    p_own, a_own = plv_aw(zm, zg)
    p_null, a_null = plv_aw(zm, zp)
    for j, prm in enumerate(PARAMS):
        rec = {'subject_id': r.subject_id, 'imaging_id': r.imaging_id, 'state': r.state, 'run': r.run,
               'session_order': r.session_order, 'mean_fd_mm': r.mean_fd_mm,
               'in_primary_analysis': r.in_primary_analysis, 'param': prm, 'n_null': len(keys)}
        same = np.array([STATE[k] == r.state for k in keys])
        for name, own, nul in (('plv', p_own[j, 0], p_null[j]), ('awplv', a_own[j, 0], a_null[j])):
            rec.update({name: own, f'{name}_null_median': np.median(nul), f'd{name}': own - np.median(nul),
                        f'{name}_p_run': (1 + np.sum(nul >= own)) / (1 + len(nul)),
                        f'd{name}_mn': own - nul.mean(), f'd{name}_same_state': own - nul[same].mean()})
            null_pool[prm][name].extend(nul)
            null_state[(r.state, prm)][name].extend(nul[same])
        rows.append(rec)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(BASE, 'results', 'synchrony', 'motion_egg_runs.csv'), index=False)

d['fed'] = (d.state == 'fed').astype(int)
d['run2'] = (d.run == 2).astype(int)
d['fed_first'] = (d.session_order == 'fed_first').astype(int)
LABEL = {'plv': 'PLV', 'awplv': 'awPLV'}
out = []
for metric in ('plv', 'awplv'):
    for prm in PARAMS:
        s = d[d.param == prm].groupby('subject_id')[f'd{metric}_mn'].mean()   # null mean: unbiased under the null
        w = stats.wilcoxon(s, alternative='greater')
        g = d[(d.param == prm) & d.in_primary_analysis]
        fit = fit_mixed(f'd{metric}_mn ~ fed + run2 + fed_first', g.reset_index(drop=True))
        out.append({'metric': LABEL[metric], 'param': prm, 'n_subjects': len(s),
                    'mean_empirical_minus_null': s.mean(),
                    'pct_runs_p05': 100 * (d[d.param == prm][f'{metric}_p_run'] < 0.05).mean(),
                    'p_above_chance': w.pvalue, 'fed_minus_fasted': fit.params['fed'],
                    'ci_low': fit.params['fed'] - 1.96 * fit.bse['fed'],
                    'ci_high': fit.params['fed'] + 1.96 * fit.bse['fed'], 'p_fed_vs_fasted': fit.pvalues['fed'],
                    'converged': bool(fit.converged)})
out = pd.DataFrame(out)
for c in ('p_above_chance', 'p_fed_vs_fasted'):
    out[c.replace('p_', 'q_', 1)] = out.groupby('metric')[c].transform(lambda x: multipletests(x, method='fdr_bh')[1])
out['p_above_chance_bonferroni'] = (out.p_above_chance * len(PARAMS)).clip(upper=1)
out.to_csv(os.path.join(BASE, 'results', 'aim2_motion_egg_control.csv'), index=False)

# Per-parameter tables with axis names (AFNI order: roll, pitch, yaw, dS, dL, dP)
AXIS = {'dL (left)': 'trans_x', 'dP (posterior)': 'trans_y', 'dS (superior)': 'trans_z',
      'pitch': 'rot_x', 'yaw': 'rot_y', 'roll': 'rot_z'}
os.makedirs(os.path.join(BASE, 'results', 'motion'), exist_ok=True)
pop = []
for metric in ('plv', 'awplv'):
    for prm in PARAMS:
        emp = d[d.param == prm][metric].values
        nul = np.array(null_pool[prm][metric])
        row = out[(out.param == prm) & (out.metric == LABEL[metric])].iloc[0]
        pop.append({'motion_param': AXIS[prm], 'afni_param': prm, 'metric': LABEL[metric], 'n_empirical': len(emp),
                    'n_null': len(nul), 'mean_empirical': emp.mean(), 'std_empirical': emp.std(ddof=1),
                    'mean_null': nul.mean(), 'std_null': nul.std(ddof=1), 'effect_size': emp.mean() - nul.mean(),
                    'n_subjects': row.n_subjects, 'p_value': row.p_above_chance, 'p_fdr': row.q_above_chance,
                    'p_bonferroni': row.p_above_chance_bonferroni, 'sig_uncorrected': row.p_above_chance < 0.05,
                    'sig_fdr': row.q_above_chance < 0.05, 'sig_bonferroni': row.p_above_chance_bonferroni < 0.05,
                    'test': 'paired one-sided Wilcoxon on participant means',
                    'mannwhitney_p_pooled_runs': stats.mannwhitneyu(emp, nul, alternative='greater').pvalue})
pd.DataFrame(pop).to_csv(os.path.join(BASE, 'results', 'motion', 'population_level_gutbff.csv'), index=False)
sm = pd.DataFrame(summary).rename(columns={k: f'{v}_mean_abs' for k, v in AXIS.items()})
sm = sm.merge(man[['imaging_id', 'run', 'subject_id']], on=['imaging_id', 'run'])
sm[[f'{v}_mean_abs' for v in AXIS.values()] + ['subject_id', 'imaging_id', 'state', 'run']].to_csv(
    os.path.join(BASE, 'results', 'motion', 'motion_summary_gutbff.csv'), index=False)
print(out.round(4).to_string(index=False))

# Population test within each state (null: other participants' EGG from the same state; statistic: participant
# mean of empirical minus null mean, which is unbiased under the null)
bys = []
for st in ('fasted', 'fed'):
    for metric in ('plv', 'awplv'):
        rows_ = []
        for prm in PARAMS:
            g = d[(d.param == prm) & (d.state == st)]
            sm_ = g.groupby('subject_id')[f'd{metric}_same_state'].mean()
            nul = np.array(null_state[(st, prm)][metric])
            rows_.append({'state': st, 'motion_param': AXIS[prm], 'afni_param': prm, 'metric': LABEL[metric],
                          'n_subjects': len(sm_), 'n_runs': len(g), 'n_null': len(nul),
                          'mean_empirical': g[metric].mean(), 'mean_null': nul.mean(),
                          'mean_empirical_minus_null': sm_.mean(),
                          'p_value': stats.wilcoxon(sm_, alternative='greater').pvalue})
        r_ = pd.DataFrame(rows_)
        r_['p_fdr'] = multipletests(r_.p_value, method='fdr_bh')[1]
        r_['p_bonferroni'] = (r_.p_value * len(PARAMS)).clip(upper=1)
        bys.append(r_)
bys = pd.concat(bys, ignore_index=True)
bys.to_csv(os.path.join(BASE, 'results', 'aim2_motion_egg_by_state.csv'), index=False)
for st in ('fasted', 'fed'):
    bys[bys.state == st].to_csv(os.path.join(BASE, 'results', 'motion', f'population_level_gutbff_{st}.csv'), index=False)
print(bys[['state', 'metric', 'afni_param', 'n_subjects', 'mean_empirical_minus_null', 'p_value', 'p_fdr']].round(4).to_string(index=False))

# Density figure: Translation | Rotation columns, X/Y/Z rows,
# PLV (light blue) and awPLV (deep blue), empirical (filled) vs pooled null (dashed), q from the paired test.
# AFNI axes: dL = x, dP = y, dS = z translations; pitch about x (R-L), yaw about y (A-P), roll about z (I-S).
GRID = {('Translation', 'X'): 'dL (left)', ('Translation', 'Y'): 'dP (posterior)', ('Translation', 'Z'): 'dS (superior)',
        ('Rotation', 'X'): 'pitch', ('Rotation', 'Y'): 'yaw', ('Rotation', 'Z'): 'roll'}
COL = {'plv': '#6baed6', 'awplv': '#08306b'}   # empirical: PLV light blue, awPLV deep blue; null grey


def stars(q):
    return '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else ''


FIG = os.path.join(BASE, 'figures', 'v2')
os.makedirs(FIG, exist_ok=True)
FOOT = ('Empirical distributions are filled (PLV light blue, awPLV deep blue); distributions from pairing each run with '
        '{null} (null) are grey and dashed. Translations: x = dL, y = dP, z = dS; rotations: x = pitch, y = yaw, z = roll. '
        'q: one-sided Wilcoxon signed-rank test on participant means of empirical minus null, FDR-corrected across six '
        'parameters per metric; * q < 0.05, ** q < 0.01, *** q < 0.001.')


def motion_figure(dd, pool, qof, name, label, title, caption):
    fig = plt.figure(figsize=(11, 12.6))
    outer = fig.add_gridspec(3, 2, wspace=0.25, hspace=0.45, left=0.11, right=0.98, top=0.86, bottom=0.04)
    for i, axis in enumerate(('X', 'Y', 'Z')):
        for j, kind in enumerate(('Translation', 'Rotation')):
            prm = GRID[(kind, axis)]
            inner = outer[i, j].subgridspec(1, 2, wspace=0.12)
            for k, metric in enumerate(('plv', 'awplv')):
                ax = fig.add_subplot(inner[k])
                emp = dd[dd.param == prm][metric].values
                nul = np.array(pool[prm][metric])
                xs = np.linspace(-0.1, 1.0, 400)
                ax.fill_between(xs, stats.gaussian_kde(emp)(xs), color=COL[metric], alpha=0.45, lw=0, label='Empirical')
                ax.plot(xs, stats.gaussian_kde(emp)(xs), color=COL[metric], lw=2)
                ax.plot(xs, stats.gaussian_kde(nul)(xs), color='#7a7a7a', lw=1.8, ls='--', label='Null')
                q = qof(prm, LABEL[metric])
                ax.set_title(('q < 0.001' if q < 0.001 else f'q = {q:.3f}') + f' {stars(q)}', fontsize=12)
                ax.set_xlim(-0.12, 0.95 if metric == 'plv' else 1.0)
                ax.set_xticks([0, 0.5] if metric == 'plv' else [0, 0.5, 1.0])
                ax.set_xlabel(LABEL[metric], fontsize=11)
                ax.set_ylabel('Density' if k == 0 else '', fontsize=11)
                if k == 1:
                    ax.tick_params(left=False, labelleft=False)
                ax.text(0.97, 0.95, prm.split(' ')[0], transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                        color='#666666')
                if i == 0 and j == 0:
                    ax.legend(frameon=True, fontsize=9, loc='upper right', bbox_to_anchor=(1.0, 0.85))
            if i == 0:
                fig.text(0.31 + 0.47 * j, 0.895, kind, ha='center', fontsize=22, fontweight='bold')
        fig.text(0.03, 0.73 - i * 0.29, axis, ha='center', va='center', fontsize=22, fontweight='bold', rotation=90)
    fig.suptitle(title, fontsize=17, fontweight='bold', x=0.02, ha='left', y=0.975)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}_nocap.{ext}'), dpi=250, facecolor='white', bbox_inches='tight')
    store_caption(FIG, name, label, caption)
    body = textwrap.fill(f'{label}: {caption}', width=118).replace(f'{label}:', rf'$\bf{{{label.replace(" ", "~")}:}}$', 1)
    fig.text(0.5, 0.0, body, ha='center', va='top', fontsize=11.5, linespacing=1.4)
    fig.savefig(os.path.join(FIG, f'{name}.png'), dpi=250, facecolor='white', bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


motion_figure(d, null_pool, lambda prm, m: out[(out.param == prm) & (out.metric == m)].iloc[0].q_above_chance,
              'fig9_motion_egg', 'Figure 9', 'EGG-head motion synchrony, both states',
              'Gaussian kernel density estimates (smoothed histograms) of synchronization between head motion and the '
              f'gastric rhythm (all usable runs, {d.subject_id.nunique()} participants, both states). '
              + FOOT.format(null="other participants' EGG from either state"))
for st, lab, num in (('fasted', 'Figure 10', 'fig10_motion_egg_fasted'), ('fed', 'Figure 11', 'fig11_motion_egg_fed')):
    ds = d[d.state == st]
    pool = {prm: null_state[(st, prm)] for prm in PARAMS}
    b = bys[bys.state == st]
    motion_figure(ds, pool, lambda prm, m, b=b: b[(b.afni_param == prm) & (b.metric == m)].iloc[0].p_fdr,
                  num, lab, f'EGG-head motion synchrony, {st} state only',
                  'Gaussian kernel density estimates (smoothed histograms) of synchronization between head motion and the '
                  f'gastric rhythm in the {st} state only ({len(ds) // len(PARAMS)} usable runs, {ds.subject_id.nunique()} participants). '
                  + FOOT.format(null=f"other participants' EGG from the {st} state"))
