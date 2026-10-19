"""
Primary outcomes with both synchrony metrics, PLV and amplitude-weighted PLV (awPLV), on the paired set.

Input: results/synchrony/region_plv_awplv_runs.csv (from 12_voxel_maps_plv_awplv.py), where synchrony is expressed as
empirical minus the mean of the mismatched-participant null.
  coupling strength  mean over the five region sets of the synchrony of each region's mean time course
  prominence         fraction of grey-matter voxels above their own null 95th percentile (identical for PLV to
                     the main pipeline)
Models: the same random-intercept mixed model as aim2_aim3_stats.fit (REML; run, session order, mean FD) with the
participant fixed-effects check, and Cohen's dz of the participant fed - fasted differences.

Output: results/primary_plv_awplv.csv
Usage: python 13_primary_plv_awplv.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from gutsync import fit_mixed

warnings.filterwarnings('ignore')
from config import BASE
RES = os.path.join(BASE, 'results')
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']


def fit(df, y):
    """As aim2_aim3_stats.fit: highest-likelihood fit over optimizers."""
    df = df.assign(fed=(df.state == 'fed').astype(int), run2=(df.run == 2).astype(int),
                   fed_first=(df.session_order == 'fed_first').astype(int)).reset_index(drop=True)
    f = f'{y} ~ fed + run2 + fed_first + mean_fd_mm'
    m = fit_mixed(f, df)
    method = 'best of 5 optimizers'
    fe = smf.ols(f.replace(' + fed_first', '') + ' + C(subject_id)', df).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(df.subject_id)[0]})
    sm = df.pivot_table(index='subject_id', columns='state', values=y)
    d = sm.fed - sm.fasted
    b, se = m.params['fed'], m.bse['fed']
    return {'fasted_mean': sm.fasted.mean(), 'fed_mean': sm.fed.mean(), 'estimate_fed_minus_fasted': b,
            'ci_low': b - 1.96 * se, 'ci_high': b + 1.96 * se, 'p_two_sided': m.pvalues['fed'],
            'fe_p_two_sided': fe.pvalues['fed'], 'dz': d.mean() / d.std(ddof=1), 'n_increase': int((d > 0).sum()),
            'n_subjects': len(d), 'converged': bool(m.converged), 'optimizer': method}


reg = pd.read_csv(os.path.join(RES, 'synchrony', 'region_plv_awplv_runs.csv'))
reg = reg[reg.in_primary_analysis]
keys = ['subject_id', 'imaging_id', 'state', 'run', 'session_order', 'mean_fd_mm']
grey = reg[reg.region == 'all-grey']
five = reg[reg.region.isin(FAMILIES)].groupby(keys, as_index=False)[['tc_dplv', 'tc_dawplv']].mean()
rows = []
for metric, m in (('PLV', 'plv'), ('awPLV', 'awplv')):
    rows.append({'outcome': 'Prominence (fraction of voxels)', 'metric': metric, **fit(grey, f'prom_{m}')})
    rows.append({'outcome': 'Coupling strength', 'metric': metric, **fit(five, f'tc_d{m}')})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, 'primary_plv_awplv.csv'), index=False)
pd.set_option('display.width', 200)
print(out.round(4).to_string(index=False))
