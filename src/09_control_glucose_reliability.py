"""
Follow-up analyses: prominence control, metabolic measures and reliability.

The voxel-averaged values (mean over voxels of the run's dPLV map within each region set) are used as an outcome
in the glucose and reliability analyses; their fed vs fasted and above-chance tests are in 14_followups.py.

1  Glucose (H3), paired set with glucose available:
     state-only:     y ~ fed + run2 + fed_first + FD + (1 | subject)
     glucose-only:   y ~ glucose_c + run2 + fed_first + FD + (1 | subject)
     combined:       y ~ fed + glucose_c + run2 + fed_first + FD + (1 | subject)
   glucose_c is session glucose minus the sample mean. State-only vs combined: likelihood-ratio test (ML fits on
   the same rows, df = 1) and AIC.
2  Reliability (same people measured twice). Run 1 vs run 2 within each state: ICC(2,1) absolute agreement
   (test-retest) and ICC(3,1) consistency. Fasted vs fed participant means: ICC(3,1) consistency (the state shift
   is expected) and Spearman rho. F-based 95% CIs (Shrout & Fleiss 1979; McGraw & Wong 1996).
3  Prominence control (08_prominence_control.py): chance level per run from mismatched EGGs; fed vs fasted on the
   control and on real minus control.

Outputs: results/followup_*.csv, results/followup_summary.txt
Usage: python 09_control_glucose_reliability.py
"""
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings('ignore')
from config import BASE
RES = os.path.join(BASE, 'results')
SYN = os.path.join(RES, 'synchrony')
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']
rng = np.random.default_rng(11)
lines = []


def log(s=''):
    lines.append(str(s))
    print(s)


def fit(df, y, extra='', state=True, reml=True):
    """Fit with each optimizer; keep converged fits with finite SEs and return the one with the highest
    likelihood (needed for valid likelihood-ratio tests between nested ML fits)."""
    f = f'{y} ~ ' + ('fed + ' if state else '') + 'run2 + fed_first + mean_fd_mm' + extra
    model = smf.mixedlm(f, df, groups=df.subject_id)
    good = []
    for method in ('lbfgs', 'bfgs', 'powell', 'nm', 'cg'):
        try:
            m = model.fit(reml=reml, method=method)
        except Exception:
            continue
        if m.converged and np.all(np.isfinite(m.bse.drop('Group Var', errors='ignore'))):
            good.append(m)
    if not good:
        # participant variance at the boundary (zero): the mixed model reduces to OLS; use clustered SEs
        m = smf.ols(f, df).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(df.subject_id)[0]})
        m.converged, m.fallback = True, 'OLS (participant variance = 0), clustered SE'
        return m
    best = max(good, key=lambda m: m.llf)
    best.fallback = ''
    return best


def state_row(df, y, label, extra=''):
    m = fit(df, y, extra)
    fe = smf.ols(f'{y} ~ fed + run2 + mean_fd_mm{extra} + C(subject_id)', df).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(df.subject_id)[0]})
    b, se = m.params['fed'], m.bse['fed']
    return {'outcome': label, 'n_subjects': df.subject_id.nunique(), 'n_runs': len(df),
            'estimate_fed_minus_fasted': b, 'ci_low': b - 1.96 * se, 'ci_high': b + 1.96 * se,
            'p_two_sided': m.pvalues['fed'], 'fe_p_two_sided': fe.pvalues['fed'], 'converged': bool(m.converged)}


# ------------------------------------------------------------------ run table
fam = pd.read_csv(os.path.join(SYN, 'family_runs.csv'))
runs = fam.drop_duplicates(['imaging_id', 'run'])[['subject_id', 'imaging_id', 'state', 'run', 'session_order',
                                                    'mean_fd_mm', 'in_primary_analysis']].copy()
runs['fed'] = (runs.state == 'fed').astype(int)
runs['run2'] = (runs.run == 2).astype(int)
runs['fed_first'] = (runs.session_order == 'fed_first').astype(int)
tc = fam[fam.region.isin(FAMILIES)].groupby(['imaging_id', 'run']).dplv.mean().rename('coupling_tc')
prom = fam[fam.region == 'all-grey'].set_index(['imaging_id', 'run']).frac_vox_above_null.rename('prominence')
runs = runs.join(tc, on=['imaging_id', 'run']).join(prom, on=['imaging_id', 'run'])

# voxel-averaged dPLV per region set from the run maps
famvol = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz')).dataobj).astype(int)
rebvol = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0
vox_rows = []
for r in runs.itertuples():
    v = np.asanyarray(nib.load(os.path.join(SYN, 'maps', f'{r.imaging_id}_run{r.run}_dplv.nii.gz')).dataobj)
    rec = {'imaging_id': r.imaging_id, 'run': r.run}
    for i, f in enumerate(FAMILIES, 1):
        sel = (famvol == i) & (v != 0)
        rec[f'vox_{f.replace("-", "_")}'] = float(v[sel].mean())
    rec['vox_rebollo2018'] = float(v[rebvol & (v != 0)].mean())
    vox_rows.append(rec)
runs = runs.merge(pd.DataFrame(vox_rows), on=['imaging_id', 'run'])
runs['vox_mean5'] = runs[[f'vox_{f.replace("-", "_")}' for f in FAMILIES]].mean(1)

subj = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_subjects.csv'))
g = subj.melt(id_vars='subject_id', value_vars=['glucose_fasted_mg_dl', 'glucose_fed_mg_dl'], value_name='glucose')
g['state'] = np.where(g.variable.str.contains('fasted'), 'fasted', 'fed')
runs = runs.merge(g[['subject_id', 'state', 'glucose']], on=['subject_id', 'state'], how='left')
P = runs[runs.in_primary_analysis].copy()

# ------------------------------------------------------------------ 4 prominence control (first: sets chance)
log('=' * 72)
log('PROMINENCE NEGATIVE CONTROL (mismatched EGG in place of own EGG)')
log('=' * 72)
CTRL = os.path.join(SYN, 'prominence_control_runs.csv')
ctrl_rows = []
if os.path.exists(CTRL):
    c = pd.read_csv(CTRL)
    P = P.merge(c, on=['imaging_id', 'run'], how='left')
    P['prom_minus_control'] = P.prominence - P.control_frac_mean
    for st in ('fasted', 'fed'):
        s = P[P.state == st].groupby('subject_id')[['prominence', 'control_frac_mean', 'prom_minus_control']].mean()
        w = stats.wilcoxon(s.prom_minus_control)
        log(f'{st:6s}: real {100 * s.prominence.mean():.2f}%  control (chance) {100 * s.control_frac_mean.mean():.2f}%  '
            f'real - control {100 * s.prom_minus_control.mean():+.2f} points, Wilcoxon p = {w.pvalue:.3f} (n = {len(s)})')
        ctrl_rows.append({'state': st, 'real_pct': 100 * s.prominence.mean(), 'control_pct': 100 * s.control_frac_mean.mean(),
                          'real_minus_control_pct': 100 * s.prom_minus_control.mean(), 'p_wilcoxon': w.pvalue, 'n': len(s)})
    for y, lab in (('control_frac_mean', 'control prominence (should not differ)'),
                   ('prom_minus_control', 'prominence minus control')):
        r = state_row(P, y, lab)
        log(f'Fed - fasted, {lab}: {100 * r["estimate_fed_minus_fasted"]:+.2f} points '
            f'[{100 * r["ci_low"]:.2f}, {100 * r["ci_high"]:.2f}], p = {r["p_two_sided"]:.4f}, FE p = {r["fe_p_two_sided"]:.4f}')
        ctrl_rows.append({'state': 'fed - fasted', 'outcome': lab, **r})
    bias = c[[x for x in c.columns if x.startswith('null_mean_minus_median')]].mean()
    log('Null bias of voxel dPLV (mean - median of null PLV, averaged over voxels and runs):')
    log(bias.round(4).to_string())
else:
    log('prominence_control_runs.csv not found; run 08_prominence_control.py')
pd.DataFrame(ctrl_rows).to_csv(os.path.join(RES, 'followup_prominence_control.csv'), index=False)

# ------------------------------------------------------------------ 2 glucose (H3)
log('\n' + '=' * 72)
log('GLUCOSE (H3)')
log('=' * 72)
G = P.dropna(subset=['glucose']).copy()
G['glucose_c'] = G.glucose - G.glucose.mean()
log(f'Rows with glucose: {len(G)} runs, {G.subject_id.nunique()} participants '
    f'(glucose SD across sessions {G.drop_duplicates(["subject_id", "state"]).glucose.std():.1f} mg/dL)')
grows = []
for y, lab in (('prominence', 'Prominence'), ('coupling_tc', 'Coupling strength'), ('vox_mean5', 'Voxel-averaged dPLV')):
    mg = fit(G, y, ' + glucose_c', state=False)
    m0 = fit(G, y, '', state=True, reml=False)
    m1 = fit(G, y, ' + glucose_c', state=True, reml=False)
    mc = fit(G, y, ' + glucose_c', state=True)
    lr = 2 * (m1.llf - m0.llf)
    k0, k1 = len(m0.params), len(m1.params)
    grows.append({'outcome': lab, 'n_subjects': G.subject_id.nunique(), 'n_runs': len(G),
                  'ols_fallback_in': ','.join(n for n, x in (('glucose_only', mg), ('state_ml', m0), ('state_glucose_ml', m1),
                                                             ('combined', mc)) if x.fallback),
                  'glucose_only_slope_per_10mgdl': 10 * mg.params['glucose_c'],
                  'glucose_only_p': mg.pvalues['glucose_c'],
                  'combined_glucose_slope_per_10mgdl': 10 * mc.params['glucose_c'],
                  'combined_glucose_p': mc.pvalues['glucose_c'], 'combined_state_estimate': mc.params['fed'],
                  'combined_state_p': mc.pvalues['fed'], 'lrt_chi2_df1': lr, 'lrt_p': stats.chi2.sf(lr, 1),
                  'aic_state_only': 2 * k0 - 2 * m0.llf, 'aic_state_glucose': 2 * k1 - 2 * m1.llf})
gr = pd.DataFrame(grows)
log(gr.round(4).to_string(index=False))
gr.to_csv(os.path.join(RES, 'followup_glucose.csv'), index=False)

# ------------------------------------------------------------------ 3 reliability
log('\n' + '=' * 72)
log('RELIABILITY')
log('=' * 72)


def icc(x, alpha=0.05):
    """x: n participants x k repeated measurements of the same people.
    ICC(2,1) = ICC(A,1), absolute agreement (two-way random): (MS_R - MS_E) / (MS_R + (k-1) MS_E + k (MS_C - MS_E) / n)
    ICC(3,1) = ICC(C,1), consistency (two-way mixed):        (MS_R - MS_E) / (MS_R + (k-1) MS_E)
    MS_R between-participant, MS_C between-measurement and MS_E residual mean squares. F-based 95% CIs from
    Shrout & Fleiss (1979) for ICC(3,1) and McGraw & Wong (1996) for ICC(A,1)."""
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    resid = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm
    mse = (resid ** 2).sum() / ((n - 1) * (k - 1))
    icc3 = (msr - mse) / (msr + (k - 1) * mse)
    icc2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    df1, df2 = n - 1, (n - 1) * (k - 1)
    F = msr / mse
    fl, fu = F / stats.f.ppf(1 - alpha / 2, df1, df2), F * stats.f.ppf(1 - alpha / 2, df2, df1)
    a = k * icc2 / (n * (1 - icc2))
    b = 1 + k * icc2 * (n - 1) / (n * (1 - icc2))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    fs, fi = stats.f.ppf(1 - alpha / 2, n - 1, v), stats.f.ppf(1 - alpha / 2, v, n - 1)
    return {'icc_2_1_agreement': icc2,
            'icc_2_1_ci_low': n * (msr - fs * mse) / (fs * (k * msc + (k * n - k - n) * mse) + n * msr),
            'icc_2_1_ci_high': n * (fi * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fi * msr),
            'icc_3_1_consistency': icc3, 'icc_3_1_ci_low': (fl - 1) / (fl + k - 1),
            'icc_3_1_ci_high': (fu - 1) / (fu + k - 1)}


# Within session: run 1 vs run 2, separately per state (one session per participant per state, so rows are
# independent participants). Across visits: fasted vs fed participant means (one pair per participant); the state
# shift is expected, so consistency ICC(3,1) is the stability index, with Spearman rho as a rank-based check.
rel = []
for y, lab in (('prominence', 'Prominence'), ('coupling_tc', 'Coupling strength'), ('vox_mean5', 'Voxel-averaged dPLV')):
    for st in ('fasted', 'fed'):
        w = runs[runs.state == st].pivot_table(index='subject_id', columns='run', values=y).dropna()
        rel.append({'measure': lab, 'comparison': f'run 1 vs run 2 ({st})', 'n_participants': len(w),
                    **icc(w.values)})
    s = P.pivot_table(index='subject_id', columns='state', values=y)[['fasted', 'fed']].dropna()
    rho, p_rho = stats.spearmanr(s.fasted, s.fed)
    rel.append({'measure': lab, 'comparison': 'fasted vs fed (participant means)', 'n_participants': len(s),
                **icc(s.values), 'spearman_rho': rho, 'spearman_p': p_rho})
rl = pd.DataFrame(rel)
log(rl.round(3).to_string(index=False))
rl.to_csv(os.path.join(RES, 'followup_reliability.csv'), index=False)
open(os.path.join(RES, 'followup_summary.txt'), 'w').write('\n'.join(lines) + '\n')
