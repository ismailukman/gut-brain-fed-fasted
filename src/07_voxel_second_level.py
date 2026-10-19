"""
Voxel-wise second-level analysis of EGG-BOLD coupling (counterpart of voxel_based_second_level_v2.py).

Input: run-level maps of dPLV = PLV - median null PLV (results/synchrony/maps), averaged per participant.
  A  coupling above chance: participant mean map over all passing runs (20 participants), one-sided
  B  fed - fasted: paired difference of participant state means (15 participants), two-sided
Inference: sign-flip permutation (10,000) of the one-sample t, family-wise error controlled with the
maximum statistic across voxels (as FSL randomise without TFCE). p = (1 + count) / (1 + n_perm).

Outputs: results/voxel_<test>_t.nii.gz, results/voxel_<test>_pfwe.nii.gz, results/voxel_second_level.csv
Usage: python 07_voxel_second_level.py
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd

from config import BASE
RES = os.path.join(BASE, 'results')
MAPS = os.path.join(RES, 'synchrony', 'maps')
N_PERM, CHUNK = 10000, 250
rng = np.random.default_rng(2026)

fam = pd.read_csv(os.path.join(RES, 'synchrony', 'family_runs.csv')).drop_duplicates(['imaging_id', 'run'])
ref = nib.load(os.path.join(MAPS, f'{fam.imaging_id.iloc[0]}_run{fam.run.iloc[0]}_dplv.nii.gz'))
vol = {}
for r in fam.itertuples():
    v = np.asanyarray(nib.load(os.path.join(MAPS, f'{r.imaging_id}_run{r.run}_dplv.nii.gz')).dataobj)
    vol.setdefault((r.subject_id, r.state), []).append(v)
mask = np.all(np.stack([m for vs in vol.values() for m in vs]) != 0, 0)
reb = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0


def subject_mean(sub, states):
    return np.mean([m for st in states for m in vol.get((sub, st), [])], 0)[mask]


def tstat(x):
    n = x.shape[-2]
    m = x.mean(-2)
    sd = x.std(-2, ddof=1)
    return m / (sd / np.sqrt(n))


def signflip(x, two_sided):
    """x: subjects x voxels. Returns observed t and FWE-corrected p per voxel."""
    n = len(x)
    t_obs = tstat(x)
    maxima = []
    for start in range(0, N_PERM, CHUNK):
        flips = rng.choice([-1.0, 1.0], size=(min(CHUNK, N_PERM - start), n, 1))
        tp = tstat(flips * x[None])
        maxima.append(np.abs(tp).max(1) if two_sided else tp.max(1))
    maxima = np.concatenate(maxima)
    stat = np.abs(t_obs) if two_sided else t_obs
    p_fwe = (1 + (maxima[None, :] >= stat[:, None]).sum(1)) / (1 + N_PERM)
    return t_obs, p_fwe


subs_all = sorted({s for s, _ in vol})
subs_paired = sorted(fam[fam.in_primary_analysis].subject_id.unique())
tests = {
    'coupling_above_chance': (np.stack([subject_mean(s, ['fasted', 'fed']) for s in subs_all]), False),
    'fed_minus_fasted': (np.stack([subject_mean(s, ['fed']) - subject_mean(s, ['fasted']) for s in subs_paired]), True),
}
rows = []
for name, (x, two) in tests.items():
    t, p = signflip(x, two)
    for arr, suf in ((t, 't'), (p, 'pfwe')):
        out = np.zeros(mask.shape, dtype=np.float32)
        out[mask] = arr
        nib.save(nib.Nifti1Image(out, ref.affine), os.path.join(RES, f'voxel_{name}_{suf}.nii.gz'))
    sig = np.zeros(mask.shape, bool)
    sig[mask] = p < 0.05
    rows.append({'test': name, 'n_subjects': len(x), 'n_voxels': int(mask.sum()), 'max_abs_t': float(np.abs(t).max()),
                 'min_p_fwe': float(p.min()), 'n_voxels_pfwe_05': int(sig.sum()),
                 'n_in_rebollo': int((sig & reb).sum()), 'sided': 'two' if two else 'one (positive)'})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, 'voxel_second_level.csv'), index=False)
print(res.to_string(index=False))
