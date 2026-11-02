"""
Voxel maps of the threshold-free score, used by 11_true_fraction.py to estimate the synchronized fraction.

For every grey-matter voxel of every usable run: the share of the mismatched-participant chance values that the
voxel's own-EGG PLV exceeds (ties count half; chance 0.5). Same signals and chance pairings as
09_threshold_free_prominence.py, whose grey-matter averages these maps reproduce (checked below).

Output: results/synchrony/maps_tf/<imaging_id>_run<r>_score_plv.nii.gz
Usage: python 10_voxel_score_maps.py
"""
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd

from gutsync import TR, analytic, bandpass, egg_peak, load_bold, load_egg

from config import BASE
from config import RAW as ROOT
OUT = os.path.join(BASE, 'results', 'synchrony', 'maps_tf')
os.makedirs(OUT, exist_ok=True)

man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
man = man[man.egg_pass].reset_index(drop=True)
pimg = nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz'))
parcels = np.asanyarray(pimg.dataobj).astype(int)
rebollo = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0
conn = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'conn_networks_3mm.nii.gz')).dataobj) > 0
gm_idx = np.flatnonzero(((parcels > 0) | rebollo | conn.any(-1)).ravel())
egg = {(r.imaging_id, r.run): load_egg(os.path.join(ROOT, r.egg_filtered_file)) for r in man.itertuples()}
ref = pd.read_csv(os.path.join(BASE, 'results', 'synchrony', 'threshold_free_runs.csv'))
ref = ref[ref.region == 'all-grey'].set_index(['imaging_id', 'run']).tf_plv

for k, r in enumerate(man.itertuples()):
    t0 = time.time()
    f0 = egg_peak(os.path.join(ROOT, r.egg_filtered_file))
    n_t = len(egg[(r.imaging_id, r.run)])
    y, _ = load_bold(r.bold_file, gm_idx)
    y = y[:n_t]
    valid = y.std(0) > 0
    zb = analytic(bandpass(y - y.mean(0), f0, 1 / TR))
    u = zb / np.where(np.abs(zb) > 0, np.abs(zb), 1)
    own = analytic(bandpass(egg[(r.imaging_id, r.run)], f0, 1 / TR))
    keys = [kk for kk in egg if kk[0][:9] != r.imaging_id[:9] and len(egg[kk]) >= n_t]
    pool = np.column_stack([analytic(bandpass(egg[kk][:n_t], f0, 1 / TR)) for kk in keys])
    e = np.abs(u.T @ np.conj(own / np.abs(own))) / len(u)
    nl = np.abs(u.T @ np.conj(pool / np.abs(pool))) / len(u)
    score = ((nl < e[:, None]).sum(1) + 0.5 * (nl == e[:, None]).sum(1)) / nl.shape[1]
    score = np.where(valid, score, np.nan)
    vol = np.full(parcels.size, np.nan, np.float32)
    vol[gm_idx] = score
    nib.save(nib.Nifti1Image(vol.reshape(parcels.shape), pimg.affine),
             os.path.join(OUT, f'{r.imaging_id}_run{r.run}_score_plv.nii.gz'))
    check = abs(np.nanmean(score) - ref.loc[(r.imaging_id, r.run)])
    print(f'[{k + 1}/{len(man)}] {r.imaging_id} run{r.run}: {time.time() - t0:.0f}s, '
          f'|mean - threshold_free_runs| = {check:.1e}', flush=True)
print('done')
