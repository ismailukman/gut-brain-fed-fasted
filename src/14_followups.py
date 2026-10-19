"""
Follow-up analyses of EGG-BOLD synchrony with both metrics, PLV and amplitude-weighted PLV (awPLV).
Input: run-level maps of empirical minus null-mean synchrony (results/synchrony/maps_v2, from
12_voxel_maps_plv_awplv.py) and the matching region-set summaries (region_plv_awplv_runs.csv).

A  Whole-brain cluster-based inference: sign-flip permutation of the one-sample t on participant maps, family-wise
   error controlled with the maximum cluster mass (sum of t within a cluster, 26-neighbour connectivity),
   cluster-forming p < 0.01 (primary) and p < 0.001 (check).
     above chance  participant mean over all passing runs (20 participants), one-sided
     fed - fasted  paired difference of participant state means (15 participants), two-sided
B  How the network value is formed: (1) synchrony of the network mean time course, which can cancel when voxels
   have different phase lags, (2) mean of voxel synchrony, (3) prominence (fraction of voxels above their own
   null 95th percentile). B1 compares the three for the region sets (region_plv_awplv_runs.csv); B2 applies (2) to
   the CONN networks, the region sets and the Rebollo 2018 regions.
     above chance  one-sided Wilcoxon signed-rank on participant means (prominence: minus the 5% nominal level is
                   not used; prominence is tested only for fed - fasted)
     fed - fasted  mixed model (state, run, session order, mean FD, random intercept) with FE check
   FDR within scheme and metric; the Rebollo regions are a prior map and are not corrected.
D  Power: smallest fed - fasted change detectable with 80% power at two-sided alpha 0.05 (2.8 standard errors).

Outputs (results/): cluster_second_level.csv, cluster_<metric>_<test>_p<thr>.nii.gz,
         voxelmean_networks_{runs,population,state}.csv, region_summaries_{population,state}.csv
Usage: python 14_followups.py
"""
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import ndimage, stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
from config import BASE
RES = os.path.join(BASE, 'results')
MAPS = os.path.join(RES, 'synchrony', 'maps_v2')
ATLAS = os.path.join(BASE, 'atlas')
N_PERM = 5000
METRICS = {'plv': 'PLV', 'awplv': 'awPLV'}
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']
rng = np.random.default_rng(2026)

reg = pd.read_csv(os.path.join(RES, 'synchrony', 'region_plv_awplv_runs.csv'))
runs = reg.drop_duplicates(['imaging_id', 'run'])[['subject_id', 'imaging_id', 'state', 'run', 'session_order',
                                                    'mean_fd_mm', 'in_primary_analysis']].reset_index(drop=True)
ref = nib.load(os.path.join(MAPS, f'{runs.imaging_id[0]}_run{runs.run[0]}_plv_dmean.nii.gz'))
maps = {m: {(a, b): np.asanyarray(nib.load(os.path.join(MAPS, f'{a}_run{b}_{m}_dmean.nii.gz')).dataobj).astype(float)
            for a, b in zip(runs.imaging_id, runs.run)} for m in METRICS}
mask = np.all(np.stack([v for m in METRICS for v in maps[m].values()]) != 0, 0)
print(f'{len(runs)} runs, {mask.sum()} voxels in the common mask', flush=True)


def mixed(g, y):
    """Fed - fasted from the mixed model (best likelihood over optimisers) and the FE check."""
    g = g.assign(fed=(g.state == 'fed').astype(int), run2=(g.run == 2).astype(int),
                 fed_first=(g.session_order == 'fed_first').astype(int)).reset_index(drop=True)
    f = f'{y} ~ fed + run2 + fed_first + mean_fd_mm'
    best = None
    for method in ('lbfgs', 'powell', 'nm', 'bfgs'):
        try:
            fit = smf.mixedlm(f, g, groups=g.subject_id).fit(reml=False, method=method)
        except Exception:
            continue
        if np.isfinite(fit.bse['fed']) and (best is None or fit.llf > best.llf):
            best = fit
    fe = smf.ols(f + ' + C(subject_id)', g).fit(cov_type='cluster', cov_kwds={'groups': g.subject_id})
    est = best if best is not None else smf.ols(f, g).fit(cov_type='cluster', cov_kwds={'groups': g.subject_id})
    b, se = est.params['fed'], est.bse['fed']
    return {'fed_minus_fasted': b, 'se': se, 'ci_low': b - 1.96 * se, 'ci_high': b + 1.96 * se,
            'p_two_sided': est.pvalues['fed'], 'fe_p_two_sided': fe.pvalues['fed'], 'mde80': 2.8 * se,
            'model': 'mixed' if best is not None else 'OLS clustered'}


def add_fdr(d, p, groups):
    d['q_fdr'] = np.nan
    for _, idx in d[d.scheme != 'Prior map'].groupby(groups).groups.items():
        d.loc[idx, 'q_fdr'] = multipletests(d.loc[idx, p], method='fdr_bh')[1]
    return d


# ------------------------------------------------------------------ A: cluster-based inference
def tstat(x):
    return x.mean(0) / (x.std(0, ddof=1) / np.sqrt(len(x)))


CONN26 = ndimage.generate_binary_structure(3, 3)


def clusters(tvol, thr, sign):
    lab, n = ndimage.label(sign * tvol > thr, structure=CONN26)
    if n == 0:
        return lab, np.zeros(0)
    return lab, ndimage.sum(sign * tvol, lab, index=np.arange(1, n + 1))


def to_vol(v):
    out = np.zeros(mask.shape)
    out[mask] = v
    return out


def cluster_test(x, two_sided, p_cdt):
    thr = stats.t.ppf(1 - p_cdt / (2 if two_sided else 1), len(x) - 1)
    signs = (1, -1) if two_sided else (1,)
    t_obs = to_vol(tstat(x))
    null_max = np.zeros(N_PERM)
    for i in range(N_PERM):
        tp = to_vol(tstat(rng.choice([-1.0, 1.0], size=(len(x), 1)) * x))
        null_max[i] = max(clusters(tp, thr, s)[1].max(initial=0) for s in signs)
    found = []
    for s in signs:
        lab, mass = clusters(t_obs, thr, s)
        found += [(s, lab == j, m, (1 + np.sum(null_max >= m)) / (1 + N_PERM)) for j, m in enumerate(mass, 1)]
    return thr, t_obs, sorted(found, key=lambda f: -f[2])


reb = np.asanyarray(nib.load(os.path.join(ATLAS, 'rebollo2018_3mm.nii.gz')).dataobj) > 0
famv = np.asanyarray(nib.load(os.path.join(ATLAS, 'families_3mm.nii.gz')).dataobj).astype(int)
connv = np.asanyarray(nib.load(os.path.join(ATLAS, 'conn_networks_3mm.nii.gz')).dataobj) > 0
conn_names = [l.strip() for l in open(os.path.join(ATLAS, 'conn_networks_labels.txt')) if l.strip()]
subs_all = sorted(runs.subject_id.unique())
subs_paired = sorted(runs[runs.in_primary_analysis].subject_id.unique())
rows = []
for m, mlab in METRICS.items():
    def smean(sub, states):
        rr = runs[(runs.subject_id == sub) & runs.state.isin(states)]
        return np.mean([maps[m][(a, b)][mask] for a, b in zip(rr.imaging_id, rr.run)], 0)
    tests = {'above_chance': (np.stack([smean(s, ['fasted', 'fed']) for s in subs_all]), False),
             'fed_minus_fasted': (np.stack([smean(s, ['fed']) - smean(s, ['fasted']) for s in subs_paired]), True)}
    for name, (x, two) in tests.items():
        for p_cdt in (0.01, 0.001):
            thr, t_obs, found = cluster_test(x, two, p_cdt)
            out = np.zeros(mask.shape, dtype=np.int16)
            for k, (s, cl, mass, pf) in enumerate(found, 1):
                out[cl] = k
                peak = np.unravel_index(np.argmax(np.where(cl, s * t_obs, -np.inf)), mask.shape)
                fl = famv[cl]
                hit = {}
                for j, nm in enumerate(conn_names):
                    hit[nm.split('.')[0]] = hit.get(nm.split('.')[0], 0) + int((connv[..., j] & cl).sum())
                rows.append({'metric': mlab, 'test': name, 'p_cluster_forming': p_cdt, 't_threshold': round(thr, 3),
                             'cluster': k, 'sign': 'positive' if s > 0 else 'negative', 'n_voxels': int(cl.sum()),
                             'mass': float(mass), 'p_fwe': pf, 'peak_t': float(t_obs[peak]),
                             'peak_mni': ', '.join(str(int(v)) for v in np.round(nib.affines.apply_affine(ref.affine, peak))),
                             'pct_in_rebollo': 100 * float((cl & reb).sum()) / cl.sum(),
                             'region_set_majority': FAMILIES[np.bincount(fl[fl > 0]).argmax() - 1] if (fl > 0).any() else 'none',
                             'conn_majority': max(hit, key=hit.get) if max(hit.values()) else 'none'})
            nib.save(nib.Nifti1Image(out, ref.affine), os.path.join(RES, f'cluster_{m}_{name}_p{p_cdt}.nii.gz'))
            print(f'A {mlab} {name} CDT p<{p_cdt}: {len(found)} clusters, smallest p_FWE = '
                  f'{min([f[3] for f in found], default=1.0):.4f}', flush=True)
cl_df = pd.DataFrame(rows)
cl_df.to_csv(os.path.join(RES, 'cluster_second_level.csv'), index=False)

# ------------------------------------------------------------------ B1: three summaries of the region sets
SUMMARY = {'tc': 'Mean time course', 'vox': 'Mean of voxels', 'prom': 'Prominence'}
r5 = reg[reg.region.isin(FAMILIES + ['rebollo2018'])].copy()
avg = r5[r5.region.isin(FAMILIES)].groupby(['subject_id', 'imaging_id', 'state', 'run', 'session_order', 'mean_fd_mm',
                                            'in_primary_analysis'], as_index=False).mean(numeric_only=True)
avg['region'] = 'average5'
r5 = pd.concat([r5, avg], ignore_index=True)
pop1, sta1 = [], []
for (region), g in r5.groupby('region', sort=False):
    sc = 'Prior map' if region == 'rebollo2018' else 'Average' if region == 'average5' else 'Region set'
    for m, mlab in METRICS.items():
        for s_key, s_lab in SUMMARY.items():
            col = f'{s_key}_d{m}' if s_key != 'prom' else f'prom_{m}'
            base = {'scheme': sc, 'region': region, 'metric': mlab, 'summary': s_lab}
            if s_key != 'prom':
                sm = g.groupby('subject_id')[col].mean()
                pop1.append({**base, 'n_subjects': len(sm), 'mean': sm.mean(),
                             'p_value': stats.wilcoxon(sm, alternative='greater').pvalue})
            sta1.append({**base, **mixed(g[g.in_primary_analysis], col)})
pop1 = add_fdr(pd.DataFrame(pop1), 'p_value', ['scheme', 'metric', 'summary'])
sta1 = add_fdr(pd.DataFrame(sta1), 'p_two_sided', ['scheme', 'metric', 'summary'])
pop1.loc[pop1.scheme == 'Average', 'q_fdr'] = np.nan
sta1.loc[sta1.scheme == 'Average', 'q_fdr'] = np.nan
pop1.to_csv(os.path.join(RES, 'region_summaries_population.csv'), index=False)
sta1.to_csv(os.path.join(RES, 'region_summaries_state.csv'), index=False)

# ------------------------------------------------------------------ B2: mean of voxel synchrony per network
nets = {}
for j, nm in enumerate(conn_names):
    nets.setdefault(('CONN', nm.split('.')[0]), np.zeros(mask.shape, bool))
    nets[('CONN', nm.split('.')[0])] |= connv[..., j]
for k, f in enumerate(FAMILIES, 1):
    nets[('Region set', f)] = famv == k
nets[('Prior map', 'rebollo2018')] = reb
nets = {k: v & mask for k, v in nets.items() if (v & mask).sum() >= 10}
vr = []
for r in runs.itertuples():
    for (sc, nm), msk in nets.items():
        vr.append({**{c: getattr(r, c) for c in runs.columns}, 'scheme': sc, 'network': nm, 'n_vox': int(msk.sum()),
                   **{f'd{m}': float(maps[m][(r.imaging_id, r.run)][msk].mean()) for m in METRICS}})
vr = pd.DataFrame(vr)
vr.to_csv(os.path.join(RES, 'voxelmean_networks_runs.csv'), index=False)
pop2, sta2 = [], []
for (sc, nm), g in vr.groupby(['scheme', 'network'], sort=False):
    for m, mlab in METRICS.items():
        sm = g.groupby('subject_id')[f'd{m}'].mean()
        pop2.append({'scheme': sc, 'network': nm, 'metric': mlab, 'n_vox': g.n_vox.iloc[0], 'n_subjects': len(sm),
                     'mean': sm.mean(), 'p_value': stats.wilcoxon(sm, alternative='greater').pvalue})
        sta2.append({'scheme': sc, 'network': nm, 'metric': mlab, **mixed(g[g.in_primary_analysis], f'd{m}')})
pop2 = add_fdr(pd.DataFrame(pop2), 'p_value', ['scheme', 'metric'])
sta2 = add_fdr(pd.DataFrame(sta2), 'p_two_sided', ['scheme', 'metric'])
pop2.to_csv(os.path.join(RES, 'voxelmean_networks_population.csv'), index=False)
sta2.to_csv(os.path.join(RES, 'voxelmean_networks_state.csv'), index=False)

pd.set_option('display.width', 220)
print(cl_df.sort_values('p_fwe').groupby(['metric', 'test', 'p_cluster_forming']).head(1).round(4).to_string(index=False))
print(pop1.round(4).to_string(index=False))
print(sta1[['scheme', 'region', 'metric', 'summary', 'fed_minus_fasted', 'p_two_sided', 'q_fdr', 'fe_p_two_sided', 'mde80']].round(4).to_string(index=False))
print(pop2.round(4).to_string(index=False))
print(sta2[['scheme', 'network', 'metric', 'fed_minus_fasted', 'p_two_sided', 'q_fdr', 'fe_p_two_sided']].round(4).to_string(index=False))
