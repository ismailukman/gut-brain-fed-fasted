"""
State manipulation and data quality, fed vs fasted.

Glucose, EGG usability and spectral quality, head motion, censoring, GCOR and TSNR.
Paired tests on participant means (the two runs of a visit are averaged).
Outputs: results/aim1_*.csv, results/aim1_summary.txt (figure: 15_figures_main.py)

Usage: python 04_state_and_quality.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from config import BASE
DATA = os.path.join(BASE, 'data')
RES = os.path.join(BASE, 'results')
os.makedirs(RES, exist_ok=True)

subj = pd.read_csv(os.path.join(DATA, 'gutbff_subjects.csv'))
runs = pd.read_csv(os.path.join(DATA, 'gutbff_runs.csv'))
nuis = pd.read_csv(os.path.join(DATA, 'gutbff_nuisance.csv'))
runs = runs.merge(nuis[['subject_id', 'state', 'run', 'gcor', 'tsnr']], on=['subject_id', 'state', 'run'], how='left')


def paired(df, a, b, label, unit, set_name):
    d = df[[a, b]].dropna()
    diff = d[b] - d[a]
    t = stats.ttest_rel(d[b], d[a])
    w = stats.wilcoxon(d[b], d[a]) if (diff != 0).sum() > 0 else None
    return {'measure': label, 'unit': unit, 'set': set_name, 'n_subjects': len(d),
            'fasted_mean': d[a].mean(), 'fasted_sd': d[a].std(),
            'fed_mean': d[b].mean(), 'fed_sd': d[b].std(),
            'diff_mean': diff.mean(), 'diff_sd': diff.std(),
            'cohens_dz': diff.mean() / diff.std() if diff.std() > 0 else np.nan,
            't': t.statistic, 'p_t': t.pvalue,
            'W': w.statistic if w else np.nan, 'p_wilcoxon': w.pvalue if w else np.nan}


def subject_means(r, col):
    return r.pivot_table(index='subject_id', columns='state', values=col, aggfunc='mean')


rows = []
# Glucose
rows.append(paired(subj, 'glucose_fasted_mg_dl', 'glucose_fed_mg_dl', 'Blood glucose', 'mg/dL', 'all'))
rows.append(paired(subj[subj.in_paired_analysis], 'glucose_fasted_mg_dl', 'glucose_fed_mg_dl',
                   'Blood glucose', 'mg/dL', 'paired EGG-pass'))

# Motion, nuisance and EGG quality (all subjects and paired set)
sets = {'all': runs, 'paired EGG-pass': runs[runs.in_primary_analysis]}
for set_name, r in sets.items():
    for col, label, unit in [('mean_fd_mm', 'Mean FD', 'mm'),
                             ('pct_vols_fd_gt_0p5', 'Volumes FD > 0.5 mm', '%'),
                             ('bold_pct_censored', 'BOLD volumes censored', '%'),
                             ('gcor', 'Global correlation (GCOR)', ''),
                             ('tsnr', 'Temporal SNR', ''),
                             ('egg_pnr_db', 'EGG peak-to-noise ratio', 'dB'),
                             ('egg_q', 'EGG peak Q factor', ''),
                             ('egg_peak_freq_cpm', 'EGG peak frequency', 'cpm')]:
        m = subject_means(r, col)
        rows.append(paired(m, 'fasted', 'fed', label, unit, set_name))
paired_tab = pd.DataFrame(rows)
paired_tab.to_csv(os.path.join(RES, 'aim1_paired_tests.csv'), index=False)

# EGG usability
use = []
for state in ('fasted', 'fed'):
    r = runs[runs.state == state]
    use.append({'state': state, 'runs_total': len(r), 'runs_pass': int(r.egg_pass.sum()),
                'runs_pass_clean': int((r.egg_qc == 'Pass').sum()),
                'runs_pass_noisy': int((r.egg_qc == 'Pass (noisy)').sum()),
                'pct_runs_pass': 100 * r.egg_pass.mean(),
                'subjects_total': subj.shape[0], 'subjects_pass': int(subj[f'passed_{state}'].sum()),
                'subjects_both_runs_pass': int((subj[f'egg_pass_runs_{state}'] == 2).sum())})
use = pd.DataFrame(use)
use.to_csv(os.path.join(RES, 'aim1_egg_usability.csv'), index=False)
fed_only = int((subj.passed_fed & ~subj.passed_fasted).sum())
fa_only = int((subj.passed_fasted & ~subj.passed_fed).sum())
mcnemar_p = stats.binomtest(fed_only, fed_only + fa_only, 0.5).pvalue if fed_only + fa_only else np.nan

# Demographics
def msd(s):
    return f'{s.mean():.1f} ± {s.std():.1f} (n={s.notna().sum()})'


demo = []
for name, s in [('All enrolled', subj), ('Paired EGG-pass set', subj[subj.in_paired_analysis])]:
    demo.append({'set': name, 'N': len(s),
                 'female/male': f"{(s.sex == 'F').sum()}/{(s.sex == 'M').sum()}",
                 'age (y)': msd(s.age_years), 'BMI (kg/m2)': msd(s.bmi_kg_m2),
                 'right/left handed': f"{(s.handedness == 'right').sum()}/{(s.handedness == 'left').sum()}",
                 'fasted-first/fed-first': f"{(s.session_order == 'fasted_first').sum()}/"
                                           f"{(s.session_order == 'fed_first').sum()}",
                 'inter-session (days)': msd(s.inter_session_days)})
demo = pd.DataFrame(demo)
demo.to_csv(os.path.join(RES, 'aim1_demographics.csv'), index=False)

# Text summary
def line(measure, set_name):
    r = paired_tab[(paired_tab.measure == measure) & (paired_tab.set == set_name)].iloc[0]
    return (f"  {measure} [{set_name}, n={r.n_subjects}]: fasted {r.fasted_mean:.2f} ± {r.fasted_sd:.2f}, "
            f"fed {r.fed_mean:.2f} ± {r.fed_sd:.2f} {r.unit}; diff {r.diff_mean:+.2f}, dz={r.cohens_dz:.2f}, "
            f"t p={r.p_t:.3g}, Wilcoxon p={r.p_wilcoxon:.3g}")


out = ['GUT-BFF: state manipulation and data quality', '=' * 60, '', 'Demographics',
       demo.to_string(index=False), '', 'EGG usability (manual QC; pass includes "Pass (noisy)")',
       use.to_string(index=False),
       f'  Subjects passing only fed: {fed_only}; only fasted: {fa_only}; exact McNemar p={mcnemar_p:.3g}',
       f'  Subjects with >=1 passing run in both states: {int(subj.in_paired_analysis.sum())}', '',
       'Paired fed vs fasted (subject means)']
for m in paired_tab.measure.unique():
    for s in ('all', 'paired EGG-pass'):
        if ((paired_tab.measure == m) & (paired_tab.set == s)).any():
            out.append(line(m, s))
txt = '\n'.join(out)
open(os.path.join(RES, 'aim1_summary.txt'), 'w').write(txt + '\n')
print(txt)
print(f'\nWrote results to {RES}')
