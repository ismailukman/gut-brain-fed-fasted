"""
EGG-BOLD phase synchrony for each run that passed EGG QC.

BOLD: errts (fast ANATICOR); censored TRs are filled by cubic-spline interpolation.
EGG:  filtered gastric signal resampled to the TR grid.
Both are band-passed at the run's EGG peak +/- 0.015 Hz (zero-phase FIR) and 15 TRs are
trimmed at each end. PLV and awPLV are computed per network family, parcel and voxel, and
against a null of the other subjects' EGG. Template dual regression with the CONN masks
gives network time courses and maps.
Outputs (results/synchrony/): family_runs.csv, parcel_runs.csv, conn_dualreg_runs.csv,
         maps/<imaging_id>_run<r>_dplv.nii.gz (PLV minus null median)

Usage: python 05_synchrony.py            all passing runs
       python 05_synchrony.py --test 01  one subject, with checks
"""
import argparse
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd

from gutsync import TR, analytic, bandpass, egg_peak, load_bold, load_egg, plv

from config import BASE
from config import RAW as ROOT
OUT = os.path.join(BASE, 'results', os.environ.get('GUTSYNC_OUT', 'synchrony'))
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']

ap = argparse.ArgumentParser()
ap.add_argument('--test', default=None, help='two-digit subject number to run alone')
args = ap.parse_args()

man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
man = man[man.egg_pass].reset_index(drop=True)
runs_info = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_runs.csv'))
man = man.merge(runs_info[['imaging_id', 'run', 'egg_peak_freq_hz', 'mean_fd_mm', 'session_order']],
                on=['imaging_id', 'run'], how='left')

fam_img = nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz'))
family = np.asanyarray(fam_img.dataobj).astype(int)
parcels = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz')).dataobj).astype(int)
rebollo = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0
conn = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'conn_networks_3mm.nii.gz')).dataobj) > 0
conn_names = [l.strip() for l in open(os.path.join(BASE, 'atlas', 'conn_networks_labels.txt')) if l.strip()]
gm = (parcels > 0) | rebollo | conn.any(-1)       # voxels analysed voxel-wise

gm_idx = np.flatnonzero(gm.ravel())
vox_family = family.ravel()[gm_idx]
vox_parcel = parcels.ravel()[gm_idx]
vox_rebollo = rebollo.ravel()[gm_idx]
M = conn.reshape(-1, conn.shape[-1])[gm_idx].astype(np.float64)   # V x 32 templates
M = M - M.mean(0)
DR_T = 3.5                                                         # stage-2 threshold (protocol 7.5)


if args.test:
    man = man[man.subject_id == f'243929_{args.test}'].reset_index(drop=True)
    print(f'TEST on {len(man)} runs of 243929_{args.test}')

# EGG on the TR grid for every passing run (also used as the null pool)
all_man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
all_man = all_man[all_man.egg_pass].reset_index(drop=True)
egg_tr = {(r.imaging_id, r.run): load_egg(os.path.join(ROOT, r.egg_filtered_file)) for r in all_man.itertuples()}

os.makedirs(os.path.join(OUT, 'maps'), exist_ok=True)
fam_rows, par_rows, conn_rows = [], [], []
for k, r in enumerate(man.itertuples()):
    t0 = time.time()
    f0 = egg_peak(os.path.join(ROOT, r.egg_filtered_file))
    y, n_cens = load_bold(r.bold_file, gm_idx)
    n_t = len(egg_tr[(r.imaging_id, r.run)])
    y = y[:n_t]
    std = y.std(0)
    valid = std > 0
    yb = bandpass(y - y.mean(0), f0, 1 / TR)
    zb = analytic(yb)                                            # T' x V
    ub = zb / np.where(np.abs(zb) > 0, np.abs(zb), 1)            # unit phase

    # Own EGG and null pool (other subjects), filtered in this run's band
    own = analytic(bandpass(egg_tr[(r.imaging_id, r.run)], f0, 1 / TR))
    pool_keys = [kk for kk in egg_tr if kk[0][:9] != r.imaging_id[:9] and len(egg_tr[kk]) >= n_t]
    pool = np.column_stack([analytic(bandpass(egg_tr[kk][:n_t], f0, 1 / TR)) for kk in pool_keys])
    ug_own = own / np.abs(own)
    ug_pool = pool / np.abs(pool)

    # Voxel-wise PLV
    v_plv = plv(ub, ug_own[:, None])[:, 0]
    v_null = plv(ub, ug_pool)                                    # V x Nnull
    v_med, v_p95 = np.median(v_null, 1), np.percentile(v_null, 95, axis=1)
    v_plv[~valid] = np.nan
    dmap = np.zeros(family.size, dtype=np.float32)
    dmap[gm_idx] = np.nan_to_num(v_plv - v_med)
    nib.save(nib.Nifti1Image(dmap.reshape(family.shape), fam_img.affine),
             os.path.join(OUT, 'maps', f'{r.imaging_id}_run{r.run}_dplv.nii.gz'))
    above = (v_plv > v_p95) & valid

    # Region time courses: mean of the band-passed voxels
    def region_rows(labels_vec, ids, names):
        rows = []
        for rid, name in zip(ids, names):
            sel = (labels_vec == rid) & valid
            if sel.sum() < 10:
                continue
            tc = analytic(yb[:, sel].mean(1))
            u = tc / np.abs(tc)
            p_own = plv(u[:, None], ug_own[:, None])[0, 0]
            p_null = plv(u[:, None], ug_pool)[0]
            w = np.abs(tc) * np.abs(own)
            w = w / w.sum()
            aw = np.abs(np.sum(w * u * np.conj(ug_own)))
            rows.append({'region': name, 'n_vox': int(sel.sum()), 'plv': p_own,
                         'null_median': np.median(p_null), 'null_p95': np.percentile(p_null, 95),
                         'dplv': p_own - np.median(p_null), 'awplv': aw,
                         'peak_vox_plv': np.nanmax(v_plv[sel]), 'frac_vox_above_null': above[sel].mean()})
        return rows

    base = {'subject_id': r.subject_id, 'imaging_id': r.imaging_id, 'state': r.state, 'run': r.run,
            'session_order': r.session_order, 'mean_fd_mm': r.mean_fd_mm, 'egg_qc': r.egg_qc,
            'in_primary_analysis': r.in_primary_analysis, 'n_censored': n_cens, 'n_trs_used': n_t,
            'max_censored_gap_trs': r.bold_max_censored_gap_trs, 'egg_peak_hz': f0, 'n_null': len(pool_keys)}
    for row in region_rows(vox_family, range(1, 6), FAMILIES):
        fam_rows.append({**base, **row})
    fam_rows.extend({**base, **row} for row in region_rows(vox_rebollo.astype(int), [1], ['rebollo2018']))
    fam_rows.extend({**base, **row} for row in region_rows(np.ones_like(vox_family), [1], ['all-grey']))
    # Template dual regression with the CONN masks
    yc = y - y.mean(0)
    tc = np.linalg.lstsq(M, yc.T, rcond=None)[0].T                 # stage 1: T x 32
    tc = (tc - tc.mean(0)) / tc.std(0)
    X = np.column_stack([np.ones(len(tc)), tc])
    beta = np.linalg.lstsq(X, yc, rcond=None)[0]              # stage 2: (33) x V
    dof = len(tc) - X.shape[1]
    sig2 = ((yc - X @ beta) ** 2).sum(0) / dof
    xtx_inv = np.diag(np.linalg.inv(X.T @ X))
    tmap = beta[1:] / np.sqrt(np.outer(xtx_inv[1:], sig2) + 1e-12)  # 32 x V
    tc_a = analytic(bandpass(tc, f0, 1 / TR))
    tc_u = tc_a / np.abs(tc_a)
    c_own = plv(tc_u, ug_own[:, None])[:, 0]
    c_null = plv(tc_u, ug_pool)
    for j, name in enumerate(conn_names):
        inside = M[:, j] > 0
        conn_rows.append({**base, 'network': name, 'plv': c_own[j], 'null_median': np.median(c_null[j]),
                          'null_p95': np.percentile(c_null[j], 95), 'dplv': c_own[j] - np.median(c_null[j]),
                          'extent_vox_t35': int((tmap[j] > DR_T).sum()),
                          'mean_t_in_template': float(tmap[j, inside].mean()),
                          'frac_template_above_null': float(above[inside].mean())})

    for row in region_rows(vox_parcel, range(1, 217), [f'p{i:03d}' for i in range(1, 217)]):
        par_rows.append({**base, **{kk: row[kk] for kk in ('region', 'plv', 'null_median', 'dplv')}})
    print(f'[{k + 1}/{len(man)}] {r.imaging_id} run{r.run}: censored={n_cens}, f0={f0 * 60:.2f} cpm, '
          f'null n={len(pool_keys)}, {time.time() - t0:.0f}s', flush=True)

    if args.test:
        # Checks that do not use the state labels
        fr = pd.DataFrame([x for x in fam_rows if x['imaging_id'] == r.imaging_id and x['run'] == r.run])
        print(fr[['region', 'n_vox', 'plv', 'null_median', 'null_p95', 'dplv', 'awplv',
                  'peak_vox_plv', 'frac_vox_above_null']].round(3).to_string(index=False))
        from scipy.signal import welch
        fq, pb = welch(y - y.mean(0), fs=1 / TR, nperseg=128, axis=0)
        pb = pb[:, valid].mean(1)
        fq_e, pe = welch(egg_tr[(r.imaging_id, r.run)], fs=1 / TR, nperseg=128)
        print(f'  EGG peak on TR grid: {fq_e[np.argmax(pe)] * 60:.2f} cpm (expected {f0 * 60:.2f}); '
              f'BOLD power at EGG peak / median BOLD power 0.01-0.1 Hz: '
              f'{np.interp(f0, fq, pb) / np.median(pb[(fq > 0.01) & (fq < 0.1)]):.2f}')
        print(f'  voxel null median of medians: {np.nanmedian(v_med):.3f}; voxel PLV median: {np.nanmedian(v_plv):.3f}; '
              f'fraction of voxels above own null p95: {above[valid].mean():.3f} (chance = 0.05)')

suffix = f'_test{args.test}' if args.test else ''
pd.DataFrame(fam_rows).to_csv(os.path.join(OUT, f'family_runs{suffix}.csv'), index=False)
pd.DataFrame(par_rows).to_csv(os.path.join(OUT, f'parcel_runs{suffix}.csv'), index=False)
pd.DataFrame(conn_rows).to_csv(os.path.join(OUT, f'conn_dualreg_runs{suffix}.csv'), index=False)
print('done')
