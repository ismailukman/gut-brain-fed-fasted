"""
Nuisance metrics per run from the afni_proc output: % censored, GCOR and TSNR.

Ventricle principal components are not used: 3dpc saves unit-norm vectors, so their SD is
constant across runs.
Output: data/gutbff_nuisance.csv

Usage: python 02_nuisance_metrics.py
"""
import os

import pandas as pd

from config import BASE

DATA = os.path.join(BASE, 'data')

inv = pd.read_csv(os.path.join(DATA, 'gutbff_preproc_inventory.csv'))
inv = inv[inv.output_variant == 'main']
rows = []
for r in inv.itertuples():
    rv = {}
    f = os.path.join(r.output_dir, f'out.ss_review.{r.afni_subj}.txt')
    if os.path.exists(f):
        for line in open(f):
            if ':' in line:
                k, v = line.split(':', 1)
                rv[k.strip()] = v.strip()
    rows.append({'subject_id': r.subject_id, 'state': r.state, 'run': r.run,
                 'pct_censored': r.pct_censored,
                 'gcor': float(rv.get('global correlation (GCOR)', 'nan')),
                 'tsnr': float(rv.get('TSNR average', 'nan'))})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(DATA, 'gutbff_nuisance.csv'), index=False)
print(out.describe().round(4))
