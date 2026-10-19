"""
Network-level EGG-BOLD synchrony.

Two network schemes, each built from ROIs whose mean BOLD time course is phase-locked to the EGG:
  CONN    8 ICA-derived networks (32 CONN ROIs, binary masks)
  Region  the five region sets (Schaefer-200 / Tian S1 parcels within each set; the insula split by parcel),
          plus the Rebollo et al. (2018) gastric-coupled regions
Per run: PLV and awPLV for every ROI with the run's EGG and with every passing EGG run of the other
participants (mismatched null); a network value is the mean over its ROIs, for the empirical and for each null
pair. Coupling above chance = empirical minus the mean of the null; because the own EGG is exchangeable with the
null EGGs when there is no synchrony, this difference has expectation zero under the null (the median does not).
  run level    p = (1 + #null >= empirical) / (1 + n_null)
  population   participant means of (empirical - null mean), one-sided Wilcoxon signed-rank; FDR across
               networks within scheme and metric; Bonferroni also reported
  state        fed vs fasted mixed model on the paired set (run, session order, FD, random intercept) and a
               participant fixed-effects check with clustered SEs; FDR across networks

Outputs: results/networks/network_runs.csv, population_level_networks.csv, fed_vs_fasted_networks.csv,
         null_pool.npy (figures: 16_figures_networks.py)
A network value is missing for a run when any of its regions has no voxels inside that run's coverage; such
network-runs are excluded from the statistics.

Usage: python 11_network_synchrony.py               compute and test
       python 11_network_synchrony.py --stats-only  test the saved run table
"""
import os
import sys
import time
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from gutsync import TR, analytic, bandpass, egg_peak, load_bold, load_egg, fit_mixed

warnings.filterwarnings('ignore')
from config import BASE
from config import RAW as ROOT
OUT = os.path.join(BASE, 'results', 'networks')
os.makedirs(OUT, exist_ok=True)
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']
MIN_VOX = 10

# ---------------------------------------------------------------- ROI definitions
conn = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'conn_networks_3mm.nii.gz')).dataobj) > 0
conn_names = [l.strip() for l in open(os.path.join(BASE, 'atlas', 'conn_networks_labels.txt')) if l.strip()]
family = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz')).dataobj).astype(int)
parcels = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz')).dataobj).astype(int)
rebollo = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0

rois = []          # (scheme, network, roi name, flat voxel indices)
for j, name in enumerate(conn_names):
    idx = np.flatnonzero(conn[..., j].ravel())
    if len(idx) >= MIN_VOX:
        rois.append(('CONN', name.split('.')[0], name, idx))
for k, fam in enumerate(FAMILIES, 1):
    for pid in np.unique(parcels[family == k]):
        if pid == 0:
            continue
        idx = np.flatnonzero(((family == k) & (parcels == pid)).ravel())
        if len(idx) >= MIN_VOX:
            rois.append(('Region', fam, f'{fam}_p{pid}', idx))
for pid in np.unique(parcels[rebollo]):
    idx = np.flatnonzero((rebollo & (parcels == pid)).ravel()) if pid else np.flatnonzero((rebollo & (parcels == 0)).ravel())
    if len(idx) >= MIN_VOX:
        rois.append(('Region', 'rebollo2018', f'rebollo_p{pid}', idx))
all_idx = np.unique(np.concatenate([r[3] for r in rois]))
pos = {v: i for i, v in enumerate(all_idx)}
roi_cols = [np.array([pos[v] for v in r[3]]) for r in rois]
print(f'{len(rois)} ROIs: ' + ', '.join(f'{s} {n}: {sum(1 for r in rois if r[0] == s and r[1] == n)}'
                                        for s, n in dict.fromkeys((r[0], r[1]) for r in rois)))


def unit(z):
    return z / np.abs(z)


def plv_aw(zr, zg):
    """zr: T x R ROI analytic signals, zg: T x K EGG analytic signals -> PLV and awPLV (R x K)."""
    ur, ug = unit(zr), unit(zg)
    plv = np.abs(ur.T @ np.conj(ug)) / len(zr)
    num = (np.abs(zr) * ur).T @ (np.abs(zg) * np.conj(ug))           # sum_t A_r A_g e^{i(phi_r - phi_g)}
    den = np.abs(zr).T @ np.abs(zg)                                   # sum_t A_r A_g
    return plv, np.abs(num) / den


STATS_ONLY = '--stats-only' in sys.argv
if not STATS_ONLY:
    # ---------------------------------------------------------------- per run
    man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
    man = man[man.egg_pass].reset_index(drop=True)
    info = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_runs.csv'))
    man = man.merge(info[['imaging_id', 'run', 'mean_fd_mm', 'session_order']], on=['imaging_id', 'run'])
    egg = {(r.imaging_id, r.run): load_egg(os.path.join(ROOT, r.egg_filtered_file)) for r in man.itertuples()}
    nets = list(dict.fromkeys((r[0], r[1]) for r in rois))
    net_members = {nk: [i for i, r in enumerate(rois) if (r[0], r[1]) == nk] for nk in nets}

    rows, null_pool = [], {nk: {'plv': [], 'awplv': []} for nk in nets}
    for k, r in enumerate(man.itertuples()):
        t0 = time.time()
        f0 = egg_peak(os.path.join(ROOT, r.egg_filtered_file))
        y, _ = load_bold(r.bold_file, all_idx)
        n = len(egg[(r.imaging_id, r.run)])
        y = y[:n]
        tc = np.column_stack([y[:, c][:, y[:, c].std(0) > 0].mean(1) for c in roi_cols])     # T x R ROI means
        zr = analytic(bandpass(tc - tc.mean(0), f0, 1 / TR))
        zg = analytic(bandpass(egg[(r.imaging_id, r.run)], f0, 1 / TR))[:, None]
        keys = [kk for kk in egg if kk[0][:9] != r.imaging_id[:9] and len(egg[kk]) >= n]
        zp = np.column_stack([analytic(bandpass(egg[kk][:n], f0, 1 / TR)) for kk in keys])
        p_own, a_own = plv_aw(zr, zg)
        p_nul, a_nul = plv_aw(zr, zp)
        for nk in nets:
            m = net_members[nk]
            rec = {'subject_id': r.subject_id, 'imaging_id': r.imaging_id, 'state': r.state, 'run': r.run,
                   'session_order': r.session_order, 'mean_fd_mm': r.mean_fd_mm,
                   'in_primary_analysis': r.in_primary_analysis, 'scheme': nk[0], 'network': nk[1],
                   'n_rois': len(m), 'n_null': len(keys)}
            for name, own, nul in (('plv', p_own[m, 0].mean(), p_nul[m].mean(0)), ('awplv', a_own[m, 0].mean(), a_nul[m].mean(0))):
                rec.update({name: own, f'{name}_null_mean': nul.mean(), f'{name}_null_median': np.median(nul),
                            f'd{name}': own - nul.mean(), f'{name}_p_run': (1 + np.sum(nul >= own)) / (1 + len(nul))})
                null_pool[nk][name].extend(nul)
            rows.append(rec)
        print(f'[{k + 1}/{len(man)}] {r.imaging_id} run{r.run}: {time.time() - t0:.0f}s', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'network_runs.csv'), index=False)
    np.save(os.path.join(OUT, 'null_pool.npy'), {f'{a}|{b}': v for (a, b), v in null_pool.items()}, allow_pickle=True)


d = pd.read_csv(os.path.join(OUT, 'network_runs.csv'))
null_pool = {tuple(k.split('|')): v for k, v in np.load(os.path.join(OUT, 'null_pool.npy'), allow_pickle=True).item().items()}
nets = list(dict.fromkeys(zip(d.scheme, d.network)))

# ---------------------------------------------------------------- statistics
LABEL = {'plv': 'PLV', 'awplv': 'awPLV'}
d['fed'] = (d.state == 'fed').astype(int)
d['run2'] = (d.run == 2).astype(int)
d['fed_first'] = (d.session_order == 'fed_first').astype(int)
pop, sta = [], []
for (scheme, net) in nets:
    g = d[(d.scheme == scheme) & (d.network == net)]
    for metric in ('plv', 'awplv'):
        s = g.dropna(subset=[f'd{metric}']).groupby('subject_id')[f'd{metric}'].mean()
        w = stats.wilcoxon(s, alternative='greater')
        emp, nul = g[metric].dropna().values, np.array(null_pool[(scheme, net)][metric], dtype=float)
        pop.append({'scheme': scheme, 'network': net, 'metric': LABEL[metric], 'n_subjects': len(s),
                    'n_empirical': len(emp), 'n_null': len(nul), 'mean_empirical': emp.mean(),
                    'mean_null': np.nanmean(nul), 'mean_empirical_minus_null': s.mean(),
                    'pct_runs_p05': 100 * (g[f'{metric}_p_run'] < 0.05).mean(), 'p_value': w.pvalue})
        gp = g[g.in_primary_analysis].dropna(subset=[f'd{metric}']).reset_index(drop=True)
        f = f'd{metric} ~ fed + run2 + fed_first + mean_fd_mm'
        fit = fit_mixed(f, gp)
        fe = smf.ols(f.replace(' + fed_first', '') + ' + C(subject_id)', gp).fit(
            cov_type='cluster', cov_kwds={'groups': pd.factorize(gp.subject_id)[0]})
        est = fit if fit is not None else fe
        sta.append({'scheme': scheme, 'network': net, 'metric': LABEL[metric],
                    'fed_minus_fasted': est.params['fed'], 'ci_low': est.params['fed'] - 1.96 * est.bse['fed'],
                    'ci_high': est.params['fed'] + 1.96 * est.bse['fed'], 'p_two_sided': est.pvalues['fed'],
                    'fe_p_two_sided': fe.pvalues['fed'], 'model': 'mixed' if fit is not None else 'FE only'})
pop, sta = pd.DataFrame(pop), pd.DataFrame(sta)
for df_, col in ((pop, 'p_value'), (sta, 'p_two_sided')):
    hyp = df_.network != 'rebollo2018'                      # Rebollo network is a separate a priori ROI
    df_['q_fdr'] = np.nan
    for (sc, mt), idx in df_[hyp].groupby(['scheme', 'metric']).groups.items():
        df_.loc[idx, 'q_fdr'] = multipletests(df_.loc[idx, col], method='fdr_bh')[1]
pop['p_bonferroni'] = pop.p_value * pop.groupby(['scheme', 'metric']).network.transform(
    lambda x: (x != 'rebollo2018').sum())
pop['p_bonferroni'] = pop.p_bonferroni.clip(upper=1)
pop.to_csv(os.path.join(OUT, 'population_level_networks.csv'), index=False)
sta.to_csv(os.path.join(OUT, 'fed_vs_fasted_networks.csv'), index=False)
print(pop.round(4).to_string(index=False))
print(sta.round(4).to_string(index=False))
