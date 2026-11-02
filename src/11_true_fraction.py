"""
Estimated fraction of truly synchronized grey-matter voxels per run (exploratory).

Prominence counts voxels whose PLV exceeds their own chance threshold; about 6.5% of voxels do so by chance, and
synchronized voxels whose PLV does not reach the threshold are missed. Here the fraction of truly synchronized voxels is
estimated from the whole distribution of voxel scores (the share of chance pairings each voxel's PLV exceeds; maps from
10_voxel_score_maps.py), following Storey's estimate of the share of true null hypotheses:

    pi0 = (share of voxels with score < lambda) / (share expected without synchrony),   pi1 = 1 - pi0

Without synchrony the own value is equally likely to take any rank among the m chance values, so the expected share is
the share of the m + 1 score levels r/m below lambda. Synchronized voxels rarely have low scores, so voxels below lambda
are almost all non-synchronized. lambda = 0.5 is the primary choice; 0.3 and 0.7 are checks. Individual voxels are not
classified, so the 57 to 60 chance values per voxel do not limit the estimate.

Statistics (paired set): fed vs fasted with the primary mixed model and fixed-effects check; pi1 against zero per state
(one-sided Wilcoxon on participant means).
Outputs: results/synchrony/true_fraction_runs.csv, results/true_fraction.csv
Usage: python 11_true_fraction.py
"""
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from gutsync import fit_mixed

warnings.filterwarnings('ignore')
from config import BASE
RES = os.path.join(BASE, 'results')
LAMBDAS = (0.3, 0.5, 0.7)


def fit(df, y):
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
            'n_subjects': len(d),
            'fasted_p_above_zero': stats.wilcoxon(sm.fasted, alternative='greater').pvalue,
            'fed_p_above_zero': stats.wilcoxon(sm.fed, alternative='greater').pvalue}


meta = pd.read_csv(os.path.join(RES, 'synchrony', 'threshold_free_runs.csv'))
meta = meta[meta.region == 'all-grey']
info = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_runs.csv'))[['imaging_id', 'run', 'mean_fd_mm', 'session_order']]
rows = []
for r in meta.itertuples():
    v = np.asanyarray(nib.load(os.path.join(RES, 'synchrony', 'maps_tf',
                                            f'{r.imaging_id}_run{r.run}_score_plv.nii.gz')).dataobj).ravel()
    v = v[np.isfinite(v)]
    levels = np.arange(int(r.n_null) + 1) / int(r.n_null)
    row = {'subject_id': r.subject_id, 'imaging_id': r.imaging_id, 'state': r.state, 'run': r.run,
           'in_primary_analysis': r.in_primary_analysis, 'n_vox': len(v)}
    for lam in LAMBDAS:
        row[f'pi1_l{int(lam * 10)}'] = 1 - (v < lam).mean() / (levels < lam).mean()
    rows.append(row)
runs = pd.DataFrame(rows).merge(info, on=['imaging_id', 'run'])
runs.to_csv(os.path.join(RES, 'synchrony', 'true_fraction_runs.csv'), index=False)
paired = runs[runs.in_primary_analysis]
out = pd.DataFrame([{'lambda': lam, **fit(paired, f'pi1_l{int(lam * 10)}')} for lam in LAMBDAS])
out.to_csv(os.path.join(RES, 'true_fraction.csv'), index=False)
pd.set_option('display.width', 250)
print(out.round(4).to_string(index=False))
