"""
Threshold-free version of prominence (exploratory sensitivity analysis).

Prominence counts the voxels whose PLV exceeds the 95th percentile of their own null. Here every voxel contributes
instead: its score is the share of its null values that its empirical PLV exceeds (ties count half),

    score_v = ( #{null_v < e_v} + 0.5 #{null_v = e_v} ) / m,

and the run value is the mean score over grey-matter voxels. If the participant's own EGG is exchangeable with the
mismatched EGGs (no true coupling), the own value is equally likely to take any rank among the m + 1 values, so
the expected score is exactly 0.5; no threshold is chosen and no chance level has to be estimated.

Per run (same signals as 08_voxel_maps_plv_awplv.py: errts, EGG on the TR grid, band-pass at the EGG peak +/- 0.015 Hz,
15 TRs trimmed at each end; null = every passing EGG run of the other participants), for PLV and awPLV, in all grey
matter and the five region sets. Standard prominence is recomputed as a check against region_plv_awplv_runs.csv.

Statistics: fed vs fasted with the primary mixed model and the fixed-effects check (paired set); score minus 0.5
on participant means, one-sample Wilcoxon signed-rank (both states, all participants; and per state, paired set).

Outputs: results/synchrony/threshold_free_runs.csv, results/threshold_free_prominence.csv
Usage: python 09_threshold_free_prominence.py               compute (reads every BOLD run) and test
       python 09_threshold_free_prominence.py --stats-only  test only
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

from gutsync import TR, analytic, bandpass, egg_peak, fit_mixed, load_bold, load_egg

warnings.filterwarnings('ignore')
from config import BASE
from config import RAW as ROOT
RES = os.path.join(BASE, 'results')
RUNS = os.path.join(RES, 'synchrony', 'threshold_free_runs.csv')
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']


def plv_aw(z, zg):
    """As 08_voxel_maps_plv_awplv.plv_aw: z T x V, zg T x K -> PLV and awPLV, each V x K."""
    a, ag = np.abs(z), np.abs(zg)
    u, ug = z / np.where(a > 0, a, 1), zg / ag
    plv = np.abs(u.T @ np.conj(ug)) / len(z)
    aw = np.abs(z.T @ np.conj(zg)) / np.maximum(a.T @ ag, 1e-30)
    return plv, aw


def compute():
    man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
    man = man[man.egg_pass].reset_index(drop=True)
    info = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_runs.csv'))
    man = man.merge(info[['imaging_id', 'run', 'mean_fd_mm', 'session_order']], on=['imaging_id', 'run'], how='left')
    family = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz')).dataobj).astype(int)
    parcels = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz')).dataobj).astype(int)
    rebollo = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0
    conn = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'conn_networks_3mm.nii.gz')).dataobj) > 0
    gm_idx = np.flatnonzero(((parcels > 0) | rebollo | conn.any(-1)).ravel())
    regions = {f: family.ravel()[gm_idx] == k for k, f in enumerate(FAMILIES, 1)}
    regions['all-grey'] = np.ones(len(gm_idx), bool)
    egg = {(r.imaging_id, r.run): load_egg(os.path.join(ROOT, r.egg_filtered_file)) for r in man.itertuples()}

    rows = []
    for k, r in enumerate(man.itertuples()):
        t0 = time.time()
        f0 = egg_peak(os.path.join(ROOT, r.egg_filtered_file))
        n_t = len(egg[(r.imaging_id, r.run)])
        y, _ = load_bold(r.bold_file, gm_idx)
        y = y[:n_t]
        valid = y.std(0) > 0
        zb = analytic(bandpass(y - y.mean(0), f0, 1 / TR))
        own = analytic(bandpass(egg[(r.imaging_id, r.run)], f0, 1 / TR))[:, None]
        keys = [kk for kk in egg if kk[0][:9] != r.imaging_id[:9] and len(egg[kk]) >= n_t]
        pool = np.column_stack([analytic(bandpass(egg[kk][:n_t], f0, 1 / TR)) for kk in keys])
        p_own, a_own = plv_aw(zb, own)
        p_nul, a_nul = plv_aw(zb, pool)
        vox = {}
        for name, e, nl in (('plv', p_own, p_nul), ('awplv', a_own, a_nul)):
            score = ((nl < e).sum(1) + 0.5 * (nl == e).sum(1)) / nl.shape[1]
            above = e[:, 0] > np.percentile(nl, 95, axis=1)
            vox[name] = (score, above)
        base = {'subject_id': r.subject_id, 'imaging_id': r.imaging_id, 'state': r.state, 'run': r.run,
                'session_order': r.session_order, 'mean_fd_mm': r.mean_fd_mm,
                'in_primary_analysis': r.in_primary_analysis, 'n_null': len(keys)}
        for reg, sel in regions.items():
            sel = sel & valid
            rows.append({**base, 'region': reg, 'n_vox': int(sel.sum()),
                         'tf_plv': float(vox['plv'][0][sel].mean()), 'tf_awplv': float(vox['awplv'][0][sel].mean()),
                         'prom_plv': float(vox['plv'][1][sel].mean()), 'prom_awplv': float(vox['awplv'][1][sel].mean())})
        print(f'[{k + 1}/{len(man)}] {r.imaging_id} run{r.run}: {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(RUNS, index=False)


def fit(df, y):
    """Primary model (as primary_plv_awplv.fit) with the fixed-effects check and Cohen's dz."""
    df = df.assign(fed=(df.state == 'fed').astype(int), run2=(df.run == 2).astype(int),
                   fed_first=(df.session_order == 'fed_first').astype(int)).reset_index(drop=True)
    f = f'{y} ~ fed + run2 + fed_first + mean_fd_mm'
    m = fit_mixed(f, df)
    fe = smf.ols(f.replace(' + fed_first', '') + ' + C(subject_id)', df).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(df.subject_id)[0]})
    sm = df.pivot_table(index='subject_id', columns='state', values=y)
    d = sm.fed - sm.fasted
    b, se = m.params['fed'], m.bse['fed']
    return {'fasted_mean': sm.fasted.mean(), 'fed_mean': sm.fed.mean(), 'estimate_fed_minus_fasted': b,
            'ci_low': b - 1.96 * se, 'ci_high': b + 1.96 * se, 'p_two_sided': m.pvalues['fed'],
            'fe_p_two_sided': fe.pvalues['fed'], 'dz': d.mean() / d.std(ddof=1), 'n_increase': int((d > 0).sum()),
            'n_subjects': len(d)}


def above_chance(df, y):
    """Participant means minus 0.5; one-sided Wilcoxon signed-rank (greater)."""
    v = df.groupby('subject_id')[y].mean() - 0.5
    return {'n': len(v), 'mean_minus_0.5': v.mean(), 'p_one_sided': stats.wilcoxon(v, alternative='greater').pvalue}


if __name__ == '__main__':
    if '--stats-only' not in sys.argv:
        compute()
    runs = pd.read_csv(RUNS)
    # check: standard prominence must reproduce the main pipeline
    ref = pd.read_csv(os.path.join(RES, 'synchrony', 'region_plv_awplv_runs.csv'))
    chk = runs.merge(ref, on=['imaging_id', 'run', 'region'], suffixes=('', '_ref'))
    print('max |prominence - main pipeline|:',
          float(np.abs(chk[['prom_plv', 'prom_awplv']].values - chk[['prom_plv_ref', 'prom_awplv_ref']].values).max()))
    rows = []
    paired = runs[runs.in_primary_analysis]
    for reg in ['all-grey'] + FAMILIES:
        for metric in ('plv', 'awplv'):
            for kind, y in (('threshold-free score', f'tf_{metric}'), ('prominence (P95)', f'prom_{metric}')):
                row = {'region': reg, 'metric': {'plv': 'PLV', 'awplv': 'awPLV'}[metric],
                       'measure': kind, **fit(paired[paired.region == reg], y)}
                if kind == 'threshold-free score':
                    for lab, d in (('both', runs[runs.region == reg]),
                                   ('fasted', paired[(paired.region == reg) & (paired.state == 'fasted')]),
                                   ('fed', paired[(paired.region == reg) & (paired.state == 'fed')])):
                        a = above_chance(d, y)
                        row.update({f'{lab}_minus_0.5': a['mean_minus_0.5'], f'{lab}_p_above': a['p_one_sided'],
                                    f'{lab}_n': a['n']})
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'threshold_free_prominence.csv'), index=False)
    pd.set_option('display.width', 250)
    cols = ['region', 'metric', 'measure', 'fasted_mean', 'fed_mean', 'estimate_fed_minus_fasted', 'ci_low',
            'ci_high', 'p_two_sided', 'fe_p_two_sided', 'dz', 'n_increase', 'fasted_p_above', 'fed_p_above']
    print(out[cols].round(4).to_string(index=False))
