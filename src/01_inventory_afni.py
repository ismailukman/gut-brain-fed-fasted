"""
List the afni_proc outputs for each GUT-BFF run.

Records per-run errts files, censoring and ss_review values for every output_gastric* folder,
including the older *_v1 folders.
Output: data/gutbff_preproc_inventory.csv

Usage: python 01_inventory_afni.py [path/to/afni_proc/output]   (default: GUTBFF_AFNI_DIR)
"""
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config import AFNI_DIR, BASE

V001 = sys.argv[1] if len(sys.argv) > 1 else AFNI_DIR
OUT = os.path.join(BASE, 'data', 'gutbff_preproc_inventory.csv')


def find_ci(folder, pattern):
    """Case-insensitive match of a regex against file names in folder."""
    rx = re.compile(pattern, re.IGNORECASE)
    hits = [f for f in os.listdir(folder) if rx.fullmatch(f)]
    return os.path.join(folder, sorted(hits)[0]) if hits else ''


def review_values(out, tag):
    f = find_ci(out, rf'out\.ss_review\.{tag}\.txt')
    vals = {}
    if f:
        for line in open(f):
            if ':' in line:
                k, v = line.split(':', 1)
                vals[k.strip()] = v.strip()
    return vals


rows = []
for ses_dir in sorted(glob.glob(os.path.join(V001, 'sub-*', 'ses-*'))):
    sub = os.path.basename(os.path.dirname(ses_dir)).replace('sub-', '')
    state = os.path.basename(ses_dir).replace('ses-', '')
    # Known layouts only (a recursive search is very slow over the USB drive)
    outs = sorted(glob.glob(os.path.join(ses_dir, 'gastric*', 'output_gastric*')) +
                  glob.glob(os.path.join(ses_dir, 'PreprocessedData_orig', 'gastric*', 'output_gastric*')))
    outs = [o for o in outs if os.path.isdir(o)]
    if not outs:
        rows.append({'subject_id': f'243929_{sub}', 'state': state, 'output_dir': ''})
        continue
    for out in outs:
        m = re.match(r'output_(gastric\d{2}[ab])(.*)', os.path.basename(out), re.IGNORECASE)
        tag, suffix = m.group(1), m.group(2)
        rv = review_values(out, tag)
        cen_f = find_ci(out, rf'censor_{tag}_combined_2\.1D')
        cen = np.loadtxt(cen_f) if cen_f else None
        for run in (1, 2):
            r = {'subject_id': f'243929_{sub}', 'state': state, 'run': run, 'afni_subj': tag,
                 'visit': tag[-1].upper(), 'output_variant': suffix or 'main',
                 'output_dir': out,
                 'output_mtime': datetime.fromtimestamp(os.path.getmtime(out)).strftime('%Y-%m-%d')}
            for kind in ('fanaticor', 'tproject'):
                r[f'errts_{kind}_run'] = find_ci(out, rf'errts\.{tag}\.r0{run}\.{kind}\+tlrc\.nii\.gz')
                r[f'errts_{kind}_all'] = find_ci(out, rf'errts\.{tag}\.{kind}\+tlrc\.HEAD')
            r['pb05_scale'] = find_ci(out, rf'pb05\.{tag}\.r0{run}\.scale\+tlrc\.(nii\.gz|HEAD)')
            r['xmat_nocensor'] = find_ci(out, r'X\.nocensor\.xmat\.1D')
            if cen is not None and len(cen) % 2 == 0:
                c = cen[(run - 1) * len(cen) // 2: run * len(cen) // 2]
                z = np.diff(np.concatenate([[0], (c == 0).astype(int), [0]]))
                r.update({'n_trs': len(c), 'n_censored': int((c == 0).sum()),
                          'pct_censored': round(100 * (c == 0).mean(), 2),
                          'max_censored_gap_trs': int((np.where(z == -1)[0] - np.where(z == 1)[0]).max())
                          if (c == 0).any() else 0})
            r['tsnr_brain'] = rv.get('TSNR average', '')
            r['blur_estimates_acf'] = rv.get('blur estimates (ACF)', '')
            r['afni_version'] = rv.get('AFNI version', '')
            rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f'Wrote {os.path.abspath(OUT)} ({len(df)} rows)')
