"""
Voxel and region-set EGG-BOLD synchrony with both metrics, PLV and amplitude-weighted PLV (awPLV), each expressed
as empirical minus the mean of the mismatched-participant null (unbiased under the null, unlike the median).

Per run (same signals as 05_synchrony.py: errts, EGG on the TR grid, band-pass at the EGG peak +/- 0.015 Hz, 15 TRs
trimmed at each end; null = every passing EGG run of the other participants):
  maps     results/synchrony/maps_v2/<imaging_id>_run<r>_{plv,awplv}_dmean.nii.gz
  regions  results/synchrony/region_plv_awplv_runs.csv: for the five region sets, the Rebollo 2018 regions and all
           grey matter, the metric of the region mean time course, the mean of voxel values, and prominence
           (fraction of voxels above their own null 95th percentile), for PLV and awPLV

Usage: python 12_voxel_maps_plv_awplv.py
"""
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd

from gutsync import TR, analytic, bandpass, egg_peak, load_bold, load_egg

from config import BASE
from config import RAW as ROOT
OUT = os.path.join(BASE, 'results', 'synchrony')
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']

man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
man = man[man.egg_pass].reset_index(drop=True)
info = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_runs.csv'))
man = man.merge(info[['imaging_id', 'run', 'mean_fd_mm', 'session_order']], on=['imaging_id', 'run'], how='left')

fam_img = nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz'))
family = np.asanyarray(fam_img.dataobj).astype(int)
parcels = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz')).dataobj).astype(int)
rebollo = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0
conn = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'conn_networks_3mm.nii.gz')).dataobj) > 0
gm = (parcels > 0) | rebollo | conn.any(-1)
gm_idx = np.flatnonzero(gm.ravel())
regions = {f: family.ravel()[gm_idx] == k for k, f in enumerate(FAMILIES, 1)}
regions['rebollo2018'] = rebollo.ravel()[gm_idx]
regions['all-grey'] = np.ones(len(gm_idx), bool)

egg = {(r.imaging_id, r.run): load_egg(os.path.join(ROOT, r.egg_filtered_file)) for r in man.itertuples()}


def plv_aw(z, zg):
    """z: T x V BOLD analytic signals, zg: T x K EGG analytic signals -> PLV and awPLV, each V x K.
    awPLV = |sum_t A_B A_E exp(i(phi_B - phi_E))| / sum_t A_B A_E."""
    a, ag = np.abs(z), np.abs(zg)
    u, ug = z / np.where(a > 0, a, 1), zg / ag
    plv = np.abs(u.T @ np.conj(ug)) / len(z)
    aw = np.abs(z.T @ np.conj(zg)) / np.maximum(a.T @ ag, 1e-30)
    return plv, aw


os.makedirs(os.path.join(OUT, 'maps_v2'), exist_ok=True)
rows = []
for k, r in enumerate(man.itertuples()):
    t0 = time.time()
    f0 = egg_peak(os.path.join(ROOT, r.egg_filtered_file))
    n_t = len(egg[(r.imaging_id, r.run)])
    y, n_cens = load_bold(r.bold_file, gm_idx)
    y = y[:n_t]
    valid = y.std(0) > 0
    yb = bandpass(y - y.mean(0), f0, 1 / TR)
    zb = analytic(yb)
    own = analytic(bandpass(egg[(r.imaging_id, r.run)], f0, 1 / TR))[:, None]
    keys = [kk for kk in egg if kk[0][:9] != r.imaging_id[:9] and len(egg[kk]) >= n_t]
    pool = np.column_stack([analytic(bandpass(egg[kk][:n_t], f0, 1 / TR)) for kk in keys])

    p_own, a_own = plv_aw(zb, own)
    p_nul, a_nul = plv_aw(zb, pool)
    vox = {}
    for name, e, nl in (('plv', p_own[:, 0], p_nul), ('awplv', a_own[:, 0], a_nul)):
        d = np.where(valid, e - nl.mean(1), np.nan)
        above = valid & (e > np.percentile(nl, 95, axis=1))
        vox[name] = (d, above)
        out = np.zeros(family.size, dtype=np.float32)
        out[gm_idx] = np.nan_to_num(d)
        nib.save(nib.Nifti1Image(out.reshape(family.shape), fam_img.affine),
                 os.path.join(OUT, 'maps_v2', f'{r.imaging_id}_run{r.run}_{name}_dmean.nii.gz'))

    base = {'subject_id': r.subject_id, 'imaging_id': r.imaging_id, 'state': r.state, 'run': r.run,
            'session_order': r.session_order, 'mean_fd_mm': r.mean_fd_mm,
            'in_primary_analysis': r.in_primary_analysis, 'n_null': len(keys), 'n_censored': n_cens}
    for reg, sel in regions.items():
        sel = sel & valid
        tc = analytic(yb[:, sel].mean(1))[:, None]
        tp, ta = plv_aw(tc, own)
        tpn, tan = plv_aw(tc, pool)
        rows.append({**base, 'region': reg, 'n_vox': int(sel.sum()),
                     'tc_dplv': tp[0, 0] - tpn.mean(), 'tc_dawplv': ta[0, 0] - tan.mean(),
                     'vox_dplv': float(np.nanmean(vox['plv'][0][sel])), 'vox_dawplv': float(np.nanmean(vox['awplv'][0][sel])),
                     'prom_plv': float(vox['plv'][1][sel].mean()), 'prom_awplv': float(vox['awplv'][1][sel].mean())})
    print(f'[{k + 1}/{len(man)}] {r.imaging_id} run{r.run}: {time.time() - t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, 'region_plv_awplv_runs.csv'), index=False)
print('done')
