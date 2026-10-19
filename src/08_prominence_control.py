"""
Negative control for prominence: replace each run's own EGG by mismatched EGGs.

For every passing run, the voxel PLV is computed against each EGG in the null pool (other participants'
passing runs). Each pool member k in turn plays the role of the run's own EGG, and the fraction of grey-matter
voxels whose PLV with k exceeds the 95th percentile of the remaining pool members is recorded. Averaged over k,
this gives the prominence expected without true synchrony (about 0.05). The fed - fasted test on this control
should show no difference if the real effect reflects synchrony rather than properties of the BOLD data.
Also saved: per-voxel null mean minus median, averaged within each family (bias of voxel dPLV under the null).

Output: results/synchrony/prominence_control_runs.csv
Usage: python 08_prominence_control.py
"""
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd

from gutsync import TR, analytic, bandpass, egg_peak, load_bold, load_egg, plv

from config import BASE
from config import RAW as ROOT
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']

man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
man = man[man.egg_pass].reset_index(drop=True)
family = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz')).dataobj).astype(int)
parcels = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz')).dataobj).astype(int)
rebollo = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0
conn = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'conn_networks_3mm.nii.gz')).dataobj) > 0
gm = (parcels > 0) | rebollo | conn.any(-1)                 # same voxel set as 05_synchrony.py
gm_idx = np.flatnonzero(gm.ravel())
vox_family = family.ravel()[gm_idx]

def loo_exceedance(v, q=95):
    """For each column j: v[:, j] > q-th percentile of the other columns (linear interpolation, as np.percentile).
    Uses one sort per row: removing rank r_j shifts every order statistic at or above r_j by one."""
    n = v.shape[1]
    order = np.argsort(v, axis=1)
    srt = np.take_along_axis(v, order, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(len(v), 0), axis=1)
    pos = q / 100 * (n - 2)                          # position within the n - 1 remaining values
    i0, frac = int(np.floor(pos)), pos - np.floor(pos)
    i1 = min(i0 + 1, n - 2)
    a = np.take_along_axis(srt, np.where(i0 >= rank, i0 + 1, i0), axis=1)
    b = np.take_along_axis(srt, np.where(i1 >= rank, i1 + 1, i1), axis=1)
    return v > a + frac * (b - a)


egg_tr = {(r.imaging_id, r.run): load_egg(os.path.join(ROOT, r.egg_filtered_file)) for r in man.itertuples()}
rows = []
for k, r in enumerate(man.itertuples()):
    t0 = time.time()
    f0 = egg_peak(os.path.join(ROOT, r.egg_filtered_file))
    y, _ = load_bold(r.bold_file, gm_idx)
    n_t = len(egg_tr[(r.imaging_id, r.run)])
    y = y[:n_t]
    valid = y.std(0) > 0
    zb = analytic(bandpass(y - y.mean(0), f0, 1 / TR))[:, valid]
    ub = zb / np.where(np.abs(zb) > 0, np.abs(zb), 1)
    keys = [kk for kk in egg_tr if kk[0][:9] != r.imaging_id[:9] and len(egg_tr[kk]) >= n_t]
    pool = np.column_stack([analytic(bandpass(egg_tr[kk][:n_t], f0, 1 / TR)) for kk in keys])
    v_null = plv(ub, pool / np.abs(pool))                      # V x N
    n = v_null.shape[1]
    fracs = loo_exceedance(v_null).mean(0)
    rec = {'imaging_id': r.imaging_id, 'run': r.run, 'n_null': n,
           'control_frac_mean': float(np.mean(fracs)), 'control_frac_sd': float(np.std(fracs, ddof=1))}
    # bias of voxel dPLV under the null: per voxel, mean minus median of its null PLVs, averaged over voxels
    skew = v_null.mean(1) - np.median(v_null, 1)
    vf = vox_family[valid]
    for i, fam in enumerate(FAMILIES, 1):
        rec[f'null_mean_minus_median_{fam}'] = float(skew[vf == i].mean())
    rec['null_mean_minus_median_voxel'] = float(skew.mean())
    rows.append(rec)
    print(f'[{k + 1}/{len(man)}] {r.imaging_id} run{r.run}: control {rec["control_frac_mean"]:.4f}, '
          f'{time.time() - t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv(os.path.join(BASE, 'results', 'synchrony', 'prominence_control_runs.csv'), index=False)
print('done')
