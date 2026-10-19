"""
Sections 2 and 3: fed vs fasted EGG-BOLD synchrony and its network localization.

Response: dPLV = PLV - median PLV of the mismatched-subject null.
Model: y ~ fed + run2 + fed_first + mean FD + (1 | subject), runs as observations.
Section 2: network-average dPLV, awPLV, prominence, CONN networks, sensitivity analyses.
Section 3: per-family models (BH-FDR), state x family test, Rebollo 2018 region and Dice overlap.
Outputs: results/aim2_*.csv, results/aim3_*.csv, results/aim2_aim3_summary.txt, group t-maps

Usage: python 06_primary_and_localization.py
"""
import glob
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from gutsync import fit_mixed
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
from config import BASE
SYN = os.path.join(BASE, 'results', 'synchrony')
RES = os.path.join(BASE, 'results')
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']

fam = pd.read_csv(os.path.join(SYN, 'family_runs.csv'))
conn = pd.read_csv(os.path.join(SYN, 'conn_dualreg_runs.csv'))
for d in (fam, conn):
    d['fed'] = (d.state == 'fed').astype(int)
    d['run2'] = (d.run == 2).astype(int)
    d['fed_first'] = (d.session_order == 'fed_first').astype(int)
    d['clean'] = d.egg_qc == 'Pass'
    d['low_censor'] = (d.n_censored / d.n_trs_used <= 0.20) & (d.max_censored_gap_trs <= 10)


def fit(df, y, fd=True, extra=''):
    """Random-intercept mixed model (highest-likelihood fit over optimizers, see gutsync.fit_mixed).
    Cross-check: OLS with participant fixed effects and participant-clustered SEs (order is absorbed)."""
    f = f'{y} ~ fed + run2 + fed_first' + (' + mean_fd_mm' if fd else '') + extra
    m = fit_mixed(f, df)
    method = 'best of 5 optimizers'
    b, se, p2 = m.params['fed'], m.bse['fed'], m.pvalues['fed']
    fe = smf.ols(f.replace(' + fed_first', '') + ' + C(subject_id)', df).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(df.subject_id)[0]})
    return {'converged': bool(m.converged), 'optimizer': method, 'subject_var': float(m.cov_re.iloc[0, 0]),
            'estimate_fed_minus_fasted': b, 'se': se, 'ci_low': b - 1.96 * se, 'ci_high': b + 1.96 * se,
            'p_two_sided': p2, 'p_one_sided_fed_gt': p2 / 2 if b > 0 else 1 - p2 / 2,
            'fe_estimate': fe.params['fed'], 'fe_p_two_sided': fe.pvalues['fed'],
            'n_subjects': df.subject_id.nunique(), 'n_runs': df[['imaging_id', 'run']].drop_duplicates().shape[0]}


def subj_means(df, y):
    return df.pivot_table(index='subject_id', columns='state', values=y, aggfunc='mean')


out_lines = []


def log(s=''):
    out_lines.append(s)
    print(s)


# Run-level tables
net_avg = (fam[fam.region.isin(FAMILIES)]
           .groupby(['subject_id', 'imaging_id', 'state', 'run', 'fed', 'run2', 'fed_first', 'mean_fd_mm',
                     'in_primary_analysis', 'clean', 'low_censor'], as_index=False)
           [['dplv', 'awplv', 'plv', 'null_median']].mean())
grey = fam[fam.region == 'all-grey'][['imaging_id', 'run', 'frac_vox_above_null', 'peak_vox_plv']]
net_avg = net_avg.merge(grey, on=['imaging_id', 'run'])
egg_q = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_runs.csv'))[['imaging_id', 'run', 'egg_q', 'egg_pnr_db']]
net_avg = net_avg.merge(egg_q, on=['imaging_id', 'run'], how='left')
# Run-level motion-EGG coupling (mean awPLV above chance over the six parameters), for sensitivity (f)
mot_runs = pd.read_csv(os.path.join(SYN, 'motion_egg_runs.csv'))
mot_c = mot_runs.groupby(['imaging_id', 'run'], as_index=False).dawplv.mean().rename(columns={'dawplv': 'motion_egg_awplv'})
net_avg = net_avg.merge(mot_c, on=['imaging_id', 'run'], how='left')
prim = net_avg[net_avg.in_primary_analysis]

# Section 2
log('=' * 72)
log('SECTION 2: PRIMARY PHYSIOLOGICAL OUTCOME')
log('=' * 72)
log(f'Passing runs analysed: {len(net_avg)} ({net_avg.subject_id.nunique()} subjects); '
    f'paired set: {len(prim)} runs ({prim.subject_id.nunique()} subjects)')

# Coupling above chance, both states pooled (subject means)
log('\nCoupling above the mismatched-subject null (both states pooled):')
sanity = []
for reg in FAMILIES + ['rebollo2018', 'all-grey']:
    s = fam[fam.region == reg].groupby('subject_id').dplv.mean()
    t = stats.ttest_1samp(s, 0)
    sanity.append({'region': reg, 'n_subjects': len(s), 'mean_dplv': s.mean(), 'sd': s.std(),
                   't': t.statistic, 'p_two_sided': t.pvalue})
sanity = pd.DataFrame(sanity)
sanity['q_bh'] = multipletests(sanity.p_two_sided, method='fdr_bh')[1]
log(sanity.round(4).to_string(index=False))
sanity.to_csv(os.path.join(RES, 'aim2_sanity_above_chance.csv'), index=False)

rows = []
for y, label in [('dplv', 'Network-average dPLV (primary)'), ('awplv', 'Network-average awPLV'),
                 ('frac_vox_above_null', 'Prominence: grey-matter fraction of gastric-coupled voxels'),
                 ('peak_vox_plv', 'Prominence: peak voxel PLV')]:
    rows.append({'analysis': label, 'set': 'paired, FD covariate', **fit(prim, y)})
# Sensitivity analyses
rows.append({'analysis': 'Network-average dPLV', 'set': '(a) clean Pass runs only', **fit(prim[prim.clean], 'dplv')})
rows.append({'analysis': 'Network-average dPLV', 'set': '(b) censoring <=20 % and gap <=10 TRs',
             **fit(prim[prim.low_censor], 'dplv')})
rows.append({'analysis': 'Network-average dPLV', 'set': '(c) no FD covariate', **fit(prim, 'dplv', fd=False)})
rows.append({'analysis': 'Network-average dPLV', 'set': '(d) all passing runs, 20 subjects', **fit(net_avg, 'dplv')})
y = 'frac_vox_above_null'
lab = 'Prominence: grey-matter fraction of gastric-coupled voxels'
rows.append({'analysis': lab, 'set': '(a) clean Pass runs only', **fit(prim[prim.clean], y)})
rows.append({'analysis': lab, 'set': '(b) censoring <=20 % and gap <=10 TRs', **fit(prim[prim.low_censor], y)})
rows.append({'analysis': lab, 'set': '(c) no FD covariate', **fit(prim, y, fd=False)})
rows.append({'analysis': lab, 'set': '(d) all passing runs, 20 subjects', **fit(net_avg, y)})
rows.append({'analysis': lab, 'set': '(e) EGG peak Q and PNR as covariates',
             **fit(prim, y, extra=' + egg_q + egg_pnr_db')})
rows.append({'analysis': 'Network-average dPLV', 'set': '(e) EGG peak Q and PNR as covariates',
             **fit(prim, 'dplv', extra=' + egg_q + egg_pnr_db')})
rows.append({'analysis': lab, 'set': '(f) motion-EGG coupling as covariate',
             **fit(prim, y, extra=' + motion_egg_awplv')})
rows.append({'analysis': 'Network-average dPLV', 'set': '(f) motion-EGG coupling as covariate',
             **fit(prim, 'dplv', extra=' + motion_egg_awplv')})
# (g) filter edges: synchrony recomputed with 35 TRs trimmed at each end (filter half-length)
T35 = os.path.join(BASE, 'results', 'synchrony_trim35', 'family_runs.csv')
if os.path.exists(T35):
    f35 = pd.read_csv(T35)
    f35 = f35[f35.in_primary_analysis].assign(fed=lambda x: (x.state == 'fed').astype(int),
                                               run2=lambda x: (x.run == 2).astype(int),
                                               fed_first=lambda x: (x.session_order == 'fed_first').astype(int))
    n35 = (f35[f35.region.isin(FAMILIES)].groupby(['subject_id', 'imaging_id', 'state', 'run', 'fed', 'run2',
                                                   'fed_first', 'mean_fd_mm'], as_index=False).dplv.mean()
           .merge(f35[f35.region == 'all-grey'][['imaging_id', 'run', 'frac_vox_above_null']], on=['imaging_id', 'run']))
    rows.append({'analysis': lab, 'set': '(g) 35-TR edge trim', **fit(n35, y)})
    rows.append({'analysis': 'Network-average dPLV', 'set': '(g) 35-TR edge trim', **fit(n35, 'dplv')})
sec2 = pd.DataFrame(rows)
sec2.to_csv(os.path.join(RES, 'aim2_models.csv'), index=False)
log('\nMixed models (fed - fasted):')
log(sec2[['analysis', 'set', 'n_subjects', 'n_runs', 'estimate_fed_minus_fasted', 'ci_low', 'ci_high',
          'p_one_sided_fed_gt', 'p_two_sided']].round(4).to_string(index=False))

sm = subj_means(prim, 'dplv')
log(f'\nSubject means, network-average dPLV: fasted {sm.fasted.mean():.4f} ± {sm.fasted.std():.4f}, '
    f'fed {sm.fed.mean():.4f} ± {sm.fed.std():.4f}; fed > fasted in {(sm.fed > sm.fasted).sum()}/{len(sm)} subjects; '
    f'paired Wilcoxon p = {stats.wilcoxon(sm.fed, sm.fasted).pvalue:.3f}')

for y in ('frac_vox_above_null', 'peak_vox_plv'):
    sm = subj_means(prim, y)
    log(f'Subject means, {y}: fasted {sm.fasted.mean():.4f} ± {sm.fasted.std():.4f}, fed {sm.fed.mean():.4f} ± '
        f'{sm.fed.std():.4f}; fed > fasted in {(sm.fed > sm.fasted).sum()}/{len(sm)}; '
        f'Wilcoxon p = {stats.wilcoxon(sm.fed, sm.fasted).pvalue:.3f}')

# CONN dual regression
log('\nCONN template dual regression: gastric-coupled networks (state-blind, all passing runs)')
cs = conn.groupby(['network', 'subject_id']).dplv.mean().reset_index()
coupled = []
for net, g in cs.groupby('network', sort=False):
    t = stats.ttest_1samp(g.dplv, 0)
    coupled.append({'network': net, 'mean_dplv': g.dplv.mean(), 't': t.statistic,
                    'p_one_sided_above_chance': t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2})
coupled = pd.DataFrame(coupled)
coupled['q_bh'] = multipletests(coupled.p_one_sided_above_chance, method='fdr_bh')[1]
coupled['above_chance_q05'] = coupled.q_bh < 0.05
coupled = coupled.sort_values('t', ascending=False)
log(coupled.round(4).to_string(index=False))
cprim = conn[conn.in_primary_analysis]
crow = []
for net in coupled[coupled.above_chance_q05].network:
    g = cprim[cprim.network == net]
    for y, lab in [('dplv', 'coupling dPLV'), ('extent_vox_t35', 'network extent (voxels, t>3.5)'),
                   ('frac_template_above_null', 'fraction of network voxels gastric-coupled')]:
        crow.append({'network': net, 'measure': lab, **fit(g, y)})
crow = pd.DataFrame(crow)
if len(crow):
    crow['q_bh'] = crow.groupby('measure').p_two_sided.transform(lambda p: multipletests(p, method='fdr_bh')[1])
    log('\nFed vs fasted in gastric-coupled CONN networks (paired set):')
    log(crow[['network', 'measure', 'estimate_fed_minus_fasted', 'ci_low', 'ci_high', 'p_two_sided', 'q_bh']]
        .round(4).to_string(index=False))
else:
    log('\nNo CONN network was above chance after FDR; fed vs fasted test not run.')
coupled.to_csv(os.path.join(RES, 'aim2_conn_coupled_networks.csv'), index=False)
crow.to_csv(os.path.join(RES, 'aim2_conn_fed_vs_fasted.csv'), index=False)

# Section 3
log('\n' + '=' * 72)
log('SECTION 3: NETWORK LOCALIZATION')
log('=' * 72)
fp = fam[fam.in_primary_analysis]
r3 = []
for reg in FAMILIES:
    r3.append({'region': reg, **fit(fp[fp.region == reg], 'dplv')})
r3 = pd.DataFrame(r3)
r3['q_bh_two_sided'] = multipletests(r3.p_two_sided, method='fdr_bh')[1]
r3['q_bh_one_sided'] = multipletests(r3.p_one_sided_fed_gt, method='fdr_bh')[1]
reb = fit(fp[fp.region == 'rebollo2018'], 'dplv')
r3 = pd.concat([r3, pd.DataFrame([{'region': 'rebollo2018 (prior gastric network)', **reb}])], ignore_index=True)
for reg in FAMILIES + ['rebollo2018']:
    sm = subj_means(fp[fp.region == reg], 'dplv')
    key = reg if reg in FAMILIES else 'rebollo2018 (prior gastric network)'
    r3.loc[r3.region == key, 'fasted_mean'] = sm.fasted.mean()
    r3.loc[r3.region == key, 'fed_mean'] = sm.fed.mean()
r3.to_csv(os.path.join(RES, 'aim3_family_models.csv'), index=False)
log('Per-family mixed models (dPLV, fed - fasted), BH across the five families:')
log(r3[['region', 'fasted_mean', 'fed_mean', 'estimate_fed_minus_fasted', 'ci_low', 'ci_high',
        'p_two_sided', 'q_bh_two_sided', 'p_one_sided_fed_gt', 'q_bh_one_sided']].round(4).to_string(index=False))

fi = fp[fp.region.isin(FAMILIES)].copy()
m = fit_mixed('dplv ~ fed * C(region) + run2 + fed_first + mean_fd_mm', fi, reml=True)
m0 = fit_mixed('dplv ~ fed + C(region) + run2 + fed_first + mean_fd_mm', fi, reml=False)
m1 = fit_mixed('dplv ~ fed * C(region) + run2 + fed_first + mean_fd_mm', fi, reml=False)
lr = 2 * (m1.llf - m0.llf)
p_int = stats.chi2.sf(lr, df=len(FAMILIES) - 1)
log(f'\nState x family interaction (likelihood-ratio test, df=4): chi2 = {lr:.2f}, p = {p_int:.3f}')
with open(os.path.join(RES, 'aim3_interaction.txt'), 'w') as f:
    f.write(str(m.summary()) + f'\nLRT state x family: chi2={lr:.3f}, df=4, p={p_int:.4f}\n')

# Spatial overlap with the Rebollo 2018 gastric network
maps = sorted(glob.glob(os.path.join(SYN, 'maps', '*_dplv.nii.gz')))
ref = nib.load(maps[0])
reb_mask = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0
pmeta = fp.drop_duplicates(['imaging_id', 'run'])[['subject_id', 'imaging_id', 'state', 'run']]
vol = {}
for r in pmeta.itertuples():
    vol[(r.subject_id, r.state, r.run)] = np.asanyarray(
        nib.load(os.path.join(SYN, 'maps', f'{r.imaging_id}_run{r.run}_dplv.nii.gz')).dataobj)
subs = sorted(pmeta.subject_id.unique())
state_mean = {st: np.stack([np.mean([v for (s, ss, _), v in vol.items() if s == sub and ss == st], 0)
                            for sub in subs]) for st in ('fasted', 'fed')}
analysed = np.any(np.stack(list(vol.values())) != 0, 0)
both = (state_mean['fasted'] + state_mean['fed']) / 2
t_all, p_all = stats.ttest_1samp(both, 0, axis=0)
t_diff, p_diff = stats.ttest_rel(state_mean['fed'], state_mean['fasted'], axis=0)
for name, arr in (('group_coupling_t', t_all), ('group_fed_minus_fasted_t', t_diff)):
    nib.save(nib.Nifti1Image(np.nan_to_num(arr).astype(np.float32) * analysed, ref.affine),
             os.path.join(RES, f'{name}.nii.gz'))


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum()) if (a.sum() + b.sum()) else np.nan


ov = []
for thr in (0.01, 0.05):
    sig = (p_all < thr) & (t_all > 0) & analysed
    # Null: the same map shifted at random in 3D (keeps its spatial clustering), restricted to the mask
    rng = np.random.default_rng(0)
    null = []
    while len(null) < 1000:
        sh = rng.integers(-20, 21, 3)
        if np.all(np.abs(sh) < 4):
            continue
        null.append(dice(np.roll(sig, tuple(sh), axis=(0, 1, 2)) & analysed, reb_mask & analysed))
    d = dice(sig, reb_mask & analysed)
    ov.append({'map': 'group coupling (both states), t>0', 'p_threshold_uncorrected': thr, 'null': 'random 3D shifts',
               'n_voxels': int(sig.sum()), 'dice_with_rebollo': d,
               'dice_null_mean': np.mean(null), 'p_perm': (np.sum(np.array(null) >= d) + 1) / 1001})
for st in ('fasted', 'fed'):
    t, p = stats.ttest_1samp(state_mean[st], 0, axis=0)
    sig = (p < 0.01) & (t > 0) & analysed
    ov.append({'map': f'group coupling ({st}), t>0', 'p_threshold_uncorrected': 0.01, 'n_voxels': int(sig.sum()),
               'dice_with_rebollo': dice(sig, reb_mask & analysed)})
ov = pd.DataFrame(ov)
ov.to_csv(os.path.join(RES, 'aim3_rebollo_overlap.csv'), index=False)
log('\nSpatial overlap with the Rebollo 2018 gastric network (paired set, voxel-wise one-sample t on subject means):')
log(ov.round(4).to_string(index=False))

open(os.path.join(RES, 'aim2_aim3_summary.txt'), 'w').write('\n'.join(out_lines) + '\n')
print(f'\nWrote results to {RES}')
