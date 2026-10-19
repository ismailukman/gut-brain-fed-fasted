"""
Shared signal functions for the GUT-BFF synchrony analyses.

bandpass   zero-phase Hamming FIR at f0 +/- 0.015 Hz (order 5*floor(fs/low), as in the EGG pipeline)
analytic   Hilbert analytic signal with TRIM samples removed at each end
load_egg   filtered EGG (10 Hz) resampled to the TR grid, whole TRs only
egg_peak   peak frequency (Hz) used when that EGG was filtered (stored next to it)
load_bold  errts voxel time series (T x V) with censored (zero) TRs filled by cubic spline
plv        phase-locking value between sets of unit-phase signals
"""
import os

import nibabel as nib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import filtfilt, firwin, hilbert, resample_poly

TR, FS_EGG, BW = 2.0, 10.0, 0.015
# Display names: three Yeo-7 networks, the Harvard-Oxford insula and the Tian S1 subcortical nuclei
REGION_LABELS = {'insular': 'Insula', 'sensorimotor': 'Sensorimotor network', 'salience': 'Salience network',
                 'default-mode': 'Default-mode network', 'subcortical': 'Subcortical nuclei',
                 'rebollo2018': 'Rebollo 2018 gastric-coupled regions', 'all-grey': 'All grey matter'}
TRIM = int(os.environ.get('GUTSYNC_TRIM', 15))   # TRs removed at each end after filtering


def bandpass(x, f0, fs):
    lo, hi = f0 - BW, f0 + BW
    n = 5 * int(np.floor(fs / lo))
    n += 1 - n % 2
    b = firwin(n, [lo, hi], pass_zero=False, fs=fs, window='hamming')
    return filtfilt(b, [1.0], x, axis=0, padlen=min(3 * n, x.shape[0] - 1))


def analytic(x):
    return hilbert(x, axis=0)[TRIM:-TRIM]


def load_egg(path):
    # Segments start at the first trigger. 02B_fa run1 and 07A_fa run2 end early;
    # whole TRs are kept and the BOLD is cut to the same length.
    g = np.load(path)
    step = int(FS_EGG * TR)
    n_tr = min(450, len(g) // step)
    assert n_tr >= 440, f'{path}: EGG only {len(g)} samples'
    return resample_poly(g[:n_tr * step], 1, step)


def egg_peak(path):
    # The band-pass for BOLD must match the EGG filter, so read the frequency saved with the EGG.
    d, f = os.path.split(path)
    return float(np.load(os.path.join(d, f.replace('gast_data_', 'max_freq'))))


def load_bold(bold_file, vox_idx):
    f, sel = bold_file, None
    if '[' in f:                                  # AFNI sub-brick selector, e.g. [0..449]
        f, sel = f.split('[')
        a, b = sel.rstrip(']').split('..')
        sel = slice(int(a), int(b) + 1)
    if not f.endswith('.nii.gz') and not f.endswith('.HEAD'):
        f += '.HEAD'
    data = np.asanyarray(nib.load(f).dataobj, dtype=np.float32)
    if sel is not None:
        data = data[..., sel]
    assert data.shape[-1] == 450, f'{bold_file}: {data.shape[-1]} volumes'
    y = data.reshape(-1, data.shape[-1])[vox_idx].T.astype(np.float64)
    censored = np.all(y == 0, axis=1)
    if censored.any():
        keep = ~censored
        y[censored] = CubicSpline(np.flatnonzero(keep), y[keep], axis=0)(np.flatnonzero(censored))
    return y, int(censored.sum())


def plv(zb, zg):
    """zb: T x K and zg: T x M unit-phase complex arrays -> K x M PLV."""
    return np.abs(zb.T @ np.conj(zg)) / zb.shape[0]


def store_caption(fig_dir, name, label, caption):
    """Store figure captions in figures/<set>/captions.json for reuse in documents."""
    import json
    path = os.path.join(fig_dir, 'captions.json')
    caps = json.load(open(path)) if os.path.exists(path) else {}
    caps[name] = {'label': label, 'caption': ' '.join(caption.split())}
    json.dump(caps, open(path, 'w'), indent=1, ensure_ascii=False)


def fit_mixed(formula, df, reml=True):
    """Random-intercept mixed model (participant). Every optimizer is tried and the fit with the highest
    likelihood among those with finite standard errors is returned (None if none qualifies). Taking the first
    optimizer that reports convergence is not enough: some stop at a clearly worse likelihood."""
    import statsmodels.formula.api as smf
    model = smf.mixedlm(formula, df, groups=df.subject_id)
    best = None
    for method in ('lbfgs', 'bfgs', 'powell', 'nm', 'cg'):
        try:
            m = model.fit(reml=reml, method=method)
        except Exception:
            continue
        if np.all(np.isfinite(m.bse.drop('Group Var', errors='ignore'))) and (best is None or m.llf > best.llf):
            best = m
    return best
