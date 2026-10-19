"""
Figure set v2 (alternative designs; the original figures are kept).

fig1_design            study design, example EGG and BOLD signals, phase-difference distributions
fig2_state_quality     glucose, standardized fed-fasted effects of all quality measures, EGG usability
fig3_primary_outcome   raincloud plots, voxel-wise coupling tail, sensitivity analyses
fig4_localization      network dumbbells and glass-brain projections with the Rebollo 2018 gastric-coupled regions

The example run in fig1 (243929_13B_fe run 2, sensorimotor family) was chosen for illustration because
it shows strong coupling. Output: figures/v2/*.png and *.pdf

Usage: python 15_figures_main.py
"""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy import stats

from surface_render import surface_panel
from gutsync import REGION_LABELS, TR, TRIM, analytic, bandpass, egg_peak, load_bold, load_egg, plv, store_caption

from config import BASE, TEMPLATE
from config import RAW as ROOT
RES = os.path.join(BASE, 'results')
DATA = os.path.join(BASE, 'data')
SEND = os.environ.get('GUTBFF_SEND') == '1'       # send-out version: no Rebollo 2018 regions
OUT = os.path.join(BASE, 'figures', 'send' if SEND else 'v2')
os.makedirs(OUT, exist_ok=True)

FA, FE = '#2a78d6', '#eb6834'            # fasted, fed
INK, SUB, MUTED, GRID = '#1f1f1e', '#55554f', '#9a9993', '#ebeae6'
FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12.5, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.labelsize': 12.5, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
                     'axes.titleweight': 'bold', 'axes.titlesize': 14, 'axes.titlecolor': INK, 'legend.fontsize': 11})
rng = np.random.default_rng(7)


def style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def letter(ax, s, x=-0.14, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=15, fontweight='bold', color=INK, va='bottom')


def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'


def save(fig, name, label=None, caption=None):
    """Save with an OHBM-style caption footer below the figure (bold 'Figure N:' followed by the text)."""
    for ext in ('png', 'pdf'):               # copy without the caption, for use in documents
        fig.savefig(os.path.join(OUT, f'{name}_nocap.{ext}'), dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.25)
    if caption:
        store_caption(OUT, name, label, caption)
        width = int(fig.get_size_inches()[0] * 7.6)
        body = textwrap.fill(f'{label}: {caption}', width=width)
        bold = label.replace(' ', r'\ ')
        body = body.replace(f'{label}:', rf'$\bf{{{bold}:}}$', 1)
        fig.text(0.5, -0.01, body, ha='center', va='top', fontsize=13, color=INK, linespacing=1.4)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


def raincloud(ax, sm, ylab, title, sub, ref=None, ref_lab=None):
    """Half-violin, jittered points and paired lines for subject means (columns fasted, fed)."""
    for x, col, c in ((0, 'fasted', FA), (1, 'fed', FE)):
        v = sm[col].values
        kde = stats.gaussian_kde(v)
        yy = np.linspace(v.min() - 0.25 * v.std(), v.max() + 0.25 * v.std(), 200)
        d = kde(yy) / kde(yy).max() * 0.32
        side = -1 if x == 0 else 1
        ax.fill_betweenx(yy, x + side * 0.12, x + side * (0.12 + d), color=c, alpha=0.28, lw=0)
        ax.plot(x + side * (0.12 + d), yy, color=c, lw=1.2)
    for _, r in sm.iterrows():
        up = r.fed > r.fasted
        ax.plot([0, 1], [r.fasted, r.fed], color=FE if up else FA, lw=1.1, alpha=0.35, zorder=1)
    ax.scatter(np.zeros(len(sm)), sm.fasted, s=38, color=FA, ec='white', lw=1, zorder=3)
    ax.scatter(np.ones(len(sm)), sm.fed, s=38, color=FE, ec='white', lw=1, zorder=3)
    for x, col in ((0, 'fasted'), (1, 'fed')):
        mu, se = sm[col].mean(), sm[col].std() / np.sqrt(len(sm))
        ax.errorbar(x, mu, yerr=1.96 * se, fmt='o', color=INK, ms=7, mfc='white', mew=2, elinewidth=2,
                    capsize=0, zorder=4)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.text(1.98, ref, ref_lab, fontsize=11, color=SUB, va='bottom', ha='right')
    ax.set_xticks([0, 1], ['Fasted', 'Fed'])
    for t, c in zip(ax.get_xticklabels(), (FA, FE)):
        t.set_color(c)
        t.set_fontweight('bold')
        t.set_fontsize(11)
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylabel(ylab)
    ax.set_title(title, pad=24)
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha='center', fontsize=11.5, color=SUB)
    style(ax)


# ---------------------------------------------------------------- data
subj = pd.read_csv(os.path.join(DATA, 'gutbff_subjects.csv'))
runs = pd.read_csv(os.path.join(DATA, 'gutbff_runs.csv'))
nuis = pd.read_csv(os.path.join(DATA, 'gutbff_nuisance.csv'))
runs = runs.merge(nuis[['subject_id', 'state', 'run', 'gcor', 'tsnr']], on=['subject_id', 'state', 'run'], how='left')
man = pd.read_csv(os.path.join(DATA, 'gutbff_analysis_manifest.csv'))
fam = pd.read_csv(os.path.join(RES, 'synchrony', 'family_runs.csv'))
models = pd.read_csv(os.path.join(RES, 'aim2_models.csv'))
fmods = pd.read_csv(os.path.join(RES, 'aim3_family_models.csv'))
paired = runs[runs.in_primary_analysis]


def subject_means(df, col):
    return df.pivot_table(index='subject_id', columns='state', values=col, aggfunc='mean').dropna()


# ================================================================ Figure 1: design and example
CACHE = os.path.join(RES, 'synchrony', 'fig1_example_signals.npz')
if os.path.exists(CACHE):
    c_ = np.load(CACHE)
    zb, zg, zm, f0 = c_['zb'], c_['zg'], c_['zm'], float(c_['f0'])
    own_plv, mis_plv = float(c_['own_plv']), float(c_['mis_plv'])
else:
    ex = man[(man.imaging_id == '243929_13B_fe') & (man.run == 2)].iloc[0]
    f0 = egg_peak(os.path.join(ROOT, ex.egg_filtered_file))
    famimg = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz')).dataobj)
    smi = np.flatnonzero(famimg.ravel() == 2)
    y, _ = load_bold(ex.bold_file, smi)
    egg = load_egg(os.path.join(ROOT, ex.egg_filtered_file))
    y = y[:len(egg)]
    yb = bandpass(y - y.mean(0), f0, 1 / TR)
    yb = yb[:, yb.std(0) > 0].mean(1)
    zb, zg = analytic(yb), analytic(bandpass(egg, f0, 1 / TR))
    # mismatched EGG: the other participant's passing run whose PLV is closest to this run's null median
    pool = man[man.egg_pass & (man.subject_id != ex.subject_id)]
    cand = []
    for r in pool.itertuples():
        g = load_egg(os.path.join(ROOT, r.egg_filtered_file))
        if len(g) < len(egg):
            continue
        zo = analytic(bandpass(g[:len(egg)], f0, 1 / TR))
        cand.append((plv((zb / abs(zb))[:, None], (zo / abs(zo))[:, None])[0, 0], zo))
    vals = np.array([c[0] for c in cand])
    own_plv = plv((zb / abs(zb))[:, None], (zg / abs(zg))[:, None])[0, 0]
    mis_plv, zm = cand[int(np.argmin(np.abs(vals - np.median(vals))))]
    np.savez(CACHE, zb=zb, zg=zg, zm=zm, f0=f0, own_plv=own_plv, mis_plv=mis_plv)

fig = plt.figure(figsize=(15.5, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[0.62, 1.25], width_ratios=[1.55, 1, 1, 0.95], hspace=0.45, wspace=0.38,
                      left=0.045, right=0.985, top=0.9, bottom=0.12)
# A: design timeline
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100)
ax.set_ylim(0, 10.6)
ax.axis('off')
letter(ax, 'A', x=-0.02, y=0.92)
ax.text(1, 10.5, 'Within-subject design: two morning visits, counterbalanced (14 fasted first, 11 fed first)',
        fontsize=11.5, fontweight='bold', color=INK, va='top')
for x0, lab, c, meal in ((3, 'Fasted visit', FA, 'overnight fast'), (52, 'Fed visit', FE, 'usual breakfast')):
    ax.add_patch(FancyBboxPatch((x0, 1.0), 45, 6.2, boxstyle='round,pad=0.3,rounding_size=1.2',
                                fc=c, alpha=0.08, ec=c, lw=1.6))
    ax.text(x0 + 1.5, 6.3, lab, fontsize=12, fontweight='bold', color=c, va='center')
    ax.text(x0 + 1.5, 4.9, meal, fontsize=11.5, color=SUB, va='center')
    steps = [('Blood\nglucose', 7), ('rsfMRI + EGG\nrun 1 · 15 min', 13), ('rsfMRI + EGG\nrun 2 · 15 min', 13)]
    xx = x0 + 13
    for txt, w in steps:
        ax.add_patch(FancyBboxPatch((xx, 2.0), w - 1.2, 3.4, boxstyle='round,pad=0.2,rounding_size=0.8',
                                    fc='white', ec=c, lw=1.2))
        ax.text(xx + (w - 1.2) / 2, 3.7, txt, ha='center', va='center', fontsize=11, color=INK)
        xx += w
ax.text(1, 9.1, '25 healthy adults · 3T Philips Ingenia Elition · TR = 2 s · 3-channel cutaneous EGG', fontsize=11.5, color=SUB,
        va='top')
ax.annotate('', xy=(51.5, 4.1), xytext=(48.6, 4.1), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1.2))
ax.text(50.05, 0.3, '1–14 days\napart', ha='center', fontsize=10.5, color=SUB)

# B: example signals
ax = fig.add_subplot(gs[1, 0])
t = (np.arange(len(zb)) + TRIM) * TR
w = (t >= 200) & (t <= 560)
ze = lambda v: (v - v.mean()) / v.std()
ax.plot(t[w], ze(zg.real)[w] + 2.6, color=INK, lw=1.8, label='Gastric rhythm (EGG)')
ax.plot(t[w], ze(zb.real)[w] - 2.6, color=FE, lw=1.8, label='Sensorimotor BOLD')
ax.legend(['Gastric rhythm (EGG)', 'Sensorimotor BOLD'],
          loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=11.5, handlelength=1.6)
ax.set_yticks([])
ax.set_xlabel('Time (s)')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.set_title('Example run: one participant, fed visit', loc='left', pad=30)
letter(ax, 'B', x=-0.04)

# C: phase-difference distributions with PLV and awPLV vectors, and this run's values against chance
COLM = {'PLV': '#6baed6', 'awPLV': '#08306b'}


def vectors(zx, zy):
    """Mean resultant vectors: PLV uses phase only; awPLV weights each time point by A_BOLD * A_EGG."""
    u = np.exp(1j * np.angle(zx * np.conj(zy)))
    w = np.abs(zx) * np.abs(zy)
    return np.mean(u), np.sum(w * u) / np.sum(w)


null_pairs = []                                  # this run's BOLD with every other participant's passing EGG
ex_sub = '243929_13'
for r in man[man.egg_pass & (man.subject_id != ex_sub)].itertuples():
    g_ = load_egg(os.path.join(ROOT, r.egg_filtered_file))
    if len(g_) < len(zg) + 2 * TRIM:
        continue
    zo = analytic(bandpass(g_[:len(zg) + 2 * TRIM], f0, 1 / TR))
    null_pairs.append([abs(v) for v in vectors(zb, zo)])
null_pairs = np.array(null_pairs)
own_v, mis_v = vectors(zb, zg), vectors(zb, zm)
for k, (vv, zz, lab, sub, c) in enumerate(((own_v, zg, 'Own EGG', 'phases cluster', FE),
                                           (mis_v, zm, "Another participant's EGG", 'phases scatter', MUTED))):
    ax = fig.add_subplot(gs[1, 1 + k], projection='polar')
    dphi = np.angle(zb * np.conj(zz))
    bins = np.linspace(-np.pi, np.pi, 25)
    h, _ = np.histogram(dphi, bins)
    ax.bar(bins[:-1] + np.diff(bins) / 2, 0.95 * h / h.max(), width=np.diff(bins), color=c, alpha=0.35,
           ec='white', lw=0.8)
    order = sorted((('PLV', vv[0], 5.0), ('awPLV', vv[1], 2.6)), key=lambda e: -abs(e[1]))   # longer arrow first
    for name, v, lw in order:
        ax.annotate('', xy=(np.angle(v), abs(v)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=COLM[name], lw=lw, mutation_scale=20))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75], ['0.25', '0.5', '0.75'], fontsize=9, color=SUB)
    ax.set_rlabel_position(112)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ['0', 'π/2', 'π', '−π/2'], fontsize=10.5, color=SUB)
    ax.grid(color=GRID)
    ax.set_title(f'{lab}\n{sub}', pad=22, fontsize=12, color=c if c != MUTED else SUB, fontweight='bold')
    ax.text(0.47, -0.17, f'PLV {abs(vv[0]):.2f}', transform=ax.transAxes, ha='right', fontsize=12,
            color=COLM['PLV'], fontweight='bold')
    ax.text(0.53, -0.17, f'awPLV {abs(vv[1]):.2f}', transform=ax.transAxes, ha='left', fontsize=12,
            color=COLM['awPLV'], fontweight='bold')
    if k == 0:
        letter(ax, 'C', x=-0.12, y=1.2)
ax = fig.add_subplot(gs[1, 3])
for j, name in enumerate(('PLV', 'awPLV')):
    nv = null_pairs[:, j]
    jit = (np.random.default_rng(3).random(len(nv)) - 0.5) * 0.28
    ax.scatter(j + jit, nv, s=16, color=MUTED, alpha=0.7, lw=0, zorder=2)
    ax.plot([j - 0.25, j + 0.25], [np.percentile(nv, 95)] * 2, color=SUB, lw=1.4, ls='--', zorder=3)
    own = abs(own_v[j])
    ax.scatter(j, own, s=170, marker='D', color=COLM[name], ec='white', lw=1.5, zorder=4)
    ax.text(j + 0.2, own, f'{100 * np.mean(nv < own):.0f}th\npercentile', va='center', fontsize=10, color=COLM[name])
ax.set_xticks([0, 1], ['PLV', 'awPLV'])
for tl, name in zip(ax.get_xticklabels(), ('PLV', 'awPLV')):
    tl.set_color(COLM[name]); tl.set_fontweight('bold')
ax.set_xlim(-0.55, 1.75)
ax.set_ylim(0, max(0.62, null_pairs.max() * 1.1, abs(own_v[1]) * 1.15))
ax.set_ylabel('Synchrony')
ax.set_title('Above chance?', pad=22, fontsize=12, fontweight='bold', color=INK)
ax.text(0.5, 1.02, f'own EGG (◆) vs {len(null_pairs)} other participants (●)', transform=ax.transAxes, ha='center',
        fontsize=10, color=SUB)
style(ax)
fig.suptitle('How stomach–brain synchrony is measured', fontsize=15, fontweight='bold', color=INK, x=0.05,
             ha='left', y=0.975)
save(fig, 'fig1_design', 'Figure 1',
     'Study design and synchrony measure. (A) Each participant completed a fasted and a fed morning visit in '
     'counterbalanced order; each visit included a capillary glucose test and two 15-min resting-state fMRI runs with '
     'concurrent electrogastrography (EGG). (B) Band-passed EGG (black) and sensorimotor BOLD signal (orange) from one '
     'fed run, chosen for illustration. (C) Left and middle: distribution of the BOLD minus EGG phase difference over '
     "that run (shaded rose) with the participant's own EGG and with another participant's EGG. The arrows are the mean "
     'resultant vectors; their length on the radial scale is the phase-locking value (PLV, light blue) and the '
     'amplitude-weighted PLV (awPLV, deep blue), from 0 (no synchrony) to 1 (constant phase difference). Right: the '
     "own-EGG values (diamonds) against the same BOLD signal paired with every other participant's EGG (dots, chance "
     'distribution; dashed line, 95th percentile).')
print(f'fig1: own PLV {abs(own_v[0]):.3f} awPLV {abs(own_v[1]):.3f}; mismatched PLV {abs(mis_v[0]):.3f}')

# ================================================================ Figure 2: state and quality
fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.35, 0.85], wspace=0.95, left=0.05, right=0.93, top=0.8,
                      bottom=0.14)
g = subj.set_index('subject_id')[['glucose_fasted_mg_dl', 'glucose_fed_mg_dl']].dropna()
g.columns = ['fasted', 'fed']
pg = stats.wilcoxon(g.fed, g.fasted).pvalue
ax = fig.add_subplot(gs[0])
raincloud(ax, g, 'Blood glucose (mg/dL)', 'Fed-state check: glucose',
          f'+{(g.fed - g.fasted).mean():.0f} mg/dL · n = {len(g)} · {fmt_p(pg)}')
letter(ax, 'A', x=-0.32)

# B: standardized effects (Cohen's dz with bootstrap 95 % CI)
measures = [('Blood glucose', g, 'all'),
            ('Mean framewise displacement', subject_means(paired, 'mean_fd_mm'), 'p'),
            ('BOLD volumes censored', subject_means(paired, 'bold_pct_censored'), 'p'),
            ('Global correlation (GCOR)', subject_means(paired, 'gcor'), 'p'),
            ('Temporal SNR', subject_means(paired, 'tsnr'), 'p'),
            ('EGG peak-to-noise ratio', subject_means(paired, 'egg_pnr_db'), 'p'),
            ('EGG peak sharpness (Q)', subject_means(paired, 'egg_q'), 'p'),
            ('EGG peak frequency', subject_means(paired, 'egg_peak_freq_cpm'), 'p')]
ax = fig.add_subplot(gs[1])
ax.axvspan(-0.2, 0.2, color=GRID, alpha=0.9, lw=0, zorder=0)
ax.text(0, -0.75, 'negligible (|dz| < 0.2)', ha='center', fontsize=10.5, color=SUB, va='center')
ax.axvline(0, color=MUTED, lw=1)
for i, (lab, sm, st) in enumerate(measures):
    dlt = (sm.fed - sm.fasted).values
    dz = dlt.mean() / dlt.std(ddof=1)
    boot = [(b.mean() / b.std(ddof=1)) for b in (rng.choice(dlt, len(dlt)) for _ in range(5000)) if b.std() > 0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = stats.wilcoxon(sm.fed, sm.fasted).pvalue
    yy = len(measures) - 1 - i
    c = FE if (lab == 'Blood glucose') else (INK if p < 0.05 else MUTED)
    ax.plot([lo, hi], [yy, yy], color=c, lw=2.6 if c != MUTED else 2, solid_capstyle='round')
    ax.plot(dz, yy, 'o', color=c, ms=10 if lab == 'Blood glucose' else 8, mec='white', mew=1.5, zorder=3)
    ax.text(2.55, yy, f'{fmt_p(p)}  (n = {len(sm)})', va='center', fontsize=10.5, color=c if c != MUTED else SUB)
ax.set_yticks(range(len(measures))[::-1], [m[0] for m in measures])
for tl, (lab, sm, _) in zip(ax.get_yticklabels(), measures):
    if lab == 'Blood glucose':
        tl.set_color(FE)
        tl.set_fontweight('bold')
ax.set_xlim(-1.2, 3.6)
ax.set_ylim(-1.1, len(measures) - 0.5)
ax.set_xlabel('Standardized fed − fasted difference (Cohen\'s dz, bootstrap 95% CI)')
ax.set_title('Fed vs fasted: glucose and data quality', pad=24)
ax.text(0.5, 1.02, 'Glucose: all with both values · other rows: 15 participants with usable EGG in both states',
        transform=ax.transAxes, ha='center', fontsize=11, color=SUB)
style(ax, grid='x')
letter(ax, 'B', x=-0.62)

# C: EGG usability (runs per state, with participants who have at least one usable run)
use = pd.read_csv(os.path.join(RES, 'aim1_egg_usability.csv')).set_index('state')
ax = fig.add_subplot(gs[2])
for y, st, c in ((1, 'fasted', FA), (0, 'fed', FE)):
    r = use.loc[st]
    parts = [('Pass', int(r.runs_pass_clean), c, 1.0, 'white'), ('Noisy', int(r.runs_pass_noisy), c, 0.45, INK),
             ('Fail', int(r.runs_total - r.runs_pass), GRID, 1.0, SUB)]
    x0 = 0
    for lab, n, col, al, tc in parts:
        ax.barh(y, n, left=x0, color=col, alpha=al, height=0.55, ec='white', lw=1.5)
        if n:
            ax.text(x0 + n / 2, y, f'{n}', ha='center', va='center', color=tc, fontsize=11.5, fontweight='bold')
        if y == 1:
            ax.text(x0 + n / 2, 1.42, lab, ha='center', va='bottom', fontsize=10.5, color=SUB)
        x0 += n
    ax.text(51, y, f'{r.pct_runs_pass:.0f}% usable\n{int(r.subjects_pass)} participants', va='center', fontsize=10.5,
            color=c, fontweight='bold')
ax.set_yticks([1, 0], ['Fasted', 'Fed'])
for tl, c in zip(ax.get_yticklabels(), (FA, FE)):
    tl.set_color(c); tl.set_fontweight('bold')
ax.set_xlim(0, 50)
ax.set_ylim(-0.5, 1.75)
ax.set_xlabel('EGG runs (of 50)')
ax.set_title('Usable EGG after review', pad=12)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='y', length=0)
letter(ax, 'C', x=-0.32)
fig.suptitle('State manipulation and data quality', fontsize=15, fontweight='bold', color=INK, x=0.06,
             ha='left', y=0.98)
save(fig, 'fig2_state_quality', 'Figure 2',
     'Fed-state check and data quality. (A) Capillary blood glucose at each visit; lines join the same participant and '
     'white circles show the mean with 95% CI. (B) Standardized fed minus fasted differences (Cohen\'s dz) with '
     'bootstrap 95% CIs; the grey band marks negligible differences (|dz| < 0.2). Glucose uses all participants with '
     'both values; the other measures use the 15 participants with usable EGG in both states. p values: Wilcoxon '
     'signed-rank tests. (C) EGG runs per state that passed manual review (dark), passed with noise (light) or '
     'failed (grey), with the share of usable runs and the number of participants with at least one usable run.')

# ================================================================ Figure 3: primary outcome
pf = fam[fam.in_primary_analysis]
dplv = (pf[pf.region.isin(FAMILIES)].groupby(['subject_id', 'imaging_id', 'state', 'run']).dplv.mean()
        .reset_index().pivot_table(index='subject_id', columns='state', values='dplv'))
frac = pf[pf.region == 'all-grey'].pivot_table(index='subject_id', columns='state', values='frac_vox_above_null') * 100
mm = models.set_index(['analysis', 'set'])
_ctrl = os.path.join(RES, 'synchrony', 'prominence_control_runs.csv')
CHANCE = (100 * pd.read_csv(_ctrl).merge(pf.drop_duplicates(['imaging_id', 'run'])[['imaging_id', 'run']],
                                         on=['imaging_id', 'run']).control_frac_mean.mean()
          if os.path.exists(_ctrl) else None)
P = mm.loc[('Network-average dPLV (primary)', 'paired, FD covariate')]
R = mm.loc[('Prominence: grey-matter fraction of gastric-coupled voxels', 'paired, FD covariate')]

# voxel dPLV maps: subject mean per state
maps = {}
for r in pf.drop_duplicates(['imaging_id', 'run']).itertuples():
    v = np.asanyarray(nib.load(os.path.join(RES, 'synchrony', 'maps', f'{r.imaging_id}_run{r.run}_dplv.nii.gz')).dataobj)
    maps.setdefault((r.subject_id, r.state), []).append(v)
analysed = np.any(np.stack([m for vs in maps.values() for m in vs]) != 0, 0)
vox = {st: [np.mean(maps[(s, st)], 0)[analysed] for s in sorted(dplv.index)] for st in ('fasted', 'fed')}

fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], hspace=0.42, wspace=0.62, left=0.2, right=0.95, top=0.9,
                      bottom=0.08)
ax = fig.add_subplot(gs[0, 0])
raincloud(ax, dplv, 'ΔPLV (PLV − null median)', 'Coupling strength',
          f'Δ = {P.estimate_fed_minus_fasted:+.3f} · {fmt_p(P.p_two_sided)} · n.s.', ref=0, ref_lab='null\nmedian')
letter(ax, 'A')
ax = fig.add_subplot(gs[0, 1])
raincloud(ax, frac, 'Voxels above own null 95th percentile (%)', 'Prominence',
          f'Δ = {100 * R.estimate_fed_minus_fasted:+.1f} points · {fmt_p(R.p_two_sided)}', ref=CHANCE,
          ref_lab=f'chance\n({CHANCE:.1f}%)' if CHANCE else None)
letter(ax, 'B')

ax = fig.add_subplot(gs[1, 0])
regions = [(r, REGION_LABELS[r]) for r in ['all-grey'] + FAMILIES + ([] if SEND else ['rebollo2018'])]
for i, (reg, lab) in enumerate(regions):
    sm = pf[pf.region == reg].pivot_table(index='subject_id', columns='state', values='frac_vox_above_null') * 100
    yy = len(regions) - 1 - i
    ax.plot([sm.fasted.mean(), sm.fed.mean()], [yy, yy], color=MUTED, lw=2.4, zorder=1)
    ax.plot(sm.fasted.mean(), yy, 'o', color=FA, ms=11, mec='white', mew=1.5, zorder=2)
    ax.plot(sm.fed.mean(), yy, 'o', color=FE, ms=11, mec='white', mew=1.5, zorder=2)
if CHANCE:
    ax.axvline(CHANCE, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(CHANCE + 0.05, -0.55, f'chance {CHANCE:.1f}%', ha='left', fontsize=11, color=INK)
ax.axhline(len(regions) - 1.5, color=GRID, lw=1.2)
ax.set_yticks(range(len(regions))[::-1], [r[1] for r in regions])
ax.set_ylim(-0.8, len(regions) - 0.4)
ax.set_xlabel('Prominence (% of voxels)')
ax.set_title('Prominence by region set', pad=26)
ax.text(0.30, 1.02, '● fasted', transform=ax.transAxes, ha='center', fontsize=12, color=FA)
ax.text(0.52, 1.02, '● fed', transform=ax.transAxes, ha='center', fontsize=12, color=FE)
ax.text(0.80, 1.02, '(participant means)', transform=ax.transAxes, ha='center', fontsize=11, color=SUB)
style(ax, grid='x')
letter(ax, 'C', x=-0.75)

ax = fig.add_subplot(gs[1, 1])
sets = ['paired, FD covariate', '(a) clean Pass runs only', '(b) censoring <=20 % and gap <=10 TRs',
        '(c) no FD covariate', '(e) EGG peak Q and PNR as covariates', '(f) motion-EGG coupling as covariate',
        '(g) 35-TR edge trim', '(d) all passing runs, 20 subjects']
names = ['Primary model', 'Clean EGG runs', 'Low censoring', 'No FD covariate', 'EGG quality covariates',
         'Motion-EGG covariate', 'Wider edge trim',
         'All 20 participants'] 
ax.axvline(0, color=MUTED, lw=1)
for i, st in enumerate(sets):
    yy = len(sets) - 1 - i
    for an, c, off in (('Network-average dPLV' if i else 'Network-average dPLV (primary)', INK, 0.16),
                       ('Prominence: grey-matter fraction of gastric-coupled voxels', FE, -0.16)):
        r = mm.loc[(an, st)]
        ax.plot([100 * r.ci_low, 100 * r.ci_high], [yy + off] * 2, color=c, lw=2.4, solid_capstyle='round')
        ax.plot(100 * r.estimate_fed_minus_fasted, yy + off, 'o', color=c, ms=7.5, mec='white', mew=1.3)
ax.set_yticks(range(len(sets))[::-1], names)
ax.set_xlabel('Fed − fasted, 95% CI (×100)')
ax.set_title('Robustness', pad=24)
ax.text(0.22, 1.02, '● coupling strength', transform=ax.transAxes, ha='center', fontsize=11, color=INK)
ax.text(0.80, 1.02, '● prominence', transform=ax.transAxes, ha='center', fontsize=11, color=FE)
style(ax, grid='x')
letter(ax, 'D', x=-0.5)
fig.suptitle('Feeding, coupling strength and spatial extent of stomach–brain synchrony', fontsize=17,
             fontweight='bold', color=INK, x=0.05, ha='left', y=0.98)
save(fig, 'fig3_primary_outcome', 'Figure 3',
     'Effect of feeding on stomach-brain synchrony in the 15 participants with usable EGG in both states. (A) Coupling '
     'strength: mean ΔPLV (PLV minus the median PLV obtained with other participants\' EGG) across the five region sets. '
     '(B) Prominence: share of grey-matter voxels whose PLV exceeds the 95th percentile of their own null; the dashed '
     'line is the chance level measured by substituting mismatched EGGs. In A and B, small points are participant means, '
     'lines join the same participant and white circles show the mean with 95% CI. (C) Mean prominence per region set, '
     'fasted (blue) and fed (orange); descriptive. (D) Fed minus fasted estimates (×100: percentage points for prominence, hundredths of ΔPLV for coupling strength) with 95% CIs from linear mixed '
     'models (state, run, session order and mean framewise displacement, with a random intercept per participant) for '
     'the primary model and seven sensitivity analyses.')

# ================================================================ Figure 4: localization
fig = plt.figure(figsize=(15, 7.4))
ax = fig.add_axes([0.13, 0.13, 0.28, 0.66])
fm = fmods.copy()
if SEND:
    fm = fm[~fm.region.str.startswith('rebollo')].reset_index(drop=True)
fm['label'] = [REGION_LABELS[r.split(' ')[0]].replace('Rebollo 2018 gastric', 'Rebollo 2018\ngastric') for r in fm.region]
for i, r in fm.iterrows():
    yy = len(fm) - 1 - i
    ax.plot([r.fasted_mean, r.fed_mean], [yy, yy], color=MUTED, lw=2.4, zorder=1)
    ax.plot(r.fasted_mean, yy, 'o', color=FA, ms=11, mec='white', mew=1.5, zorder=2)
    ax.plot(r.fed_mean, yy, 'o', color=FE, ms=11, mec='white', mew=1.5, zorder=2)
    q = '' if pd.isna(r.q_bh_two_sided) else f', q = {r.q_bh_two_sided:.2f}'
    ax.text(0.085, yy, f'{r.estimate_fed_minus_fasted:+.3f}\n{fmt_p(r.p_two_sided)}{q}', va='center',
            fontsize=10.5, color=INK if r.p_two_sided < 0.1 else SUB)
ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.axhline(0.5, color=GRID, lw=1.2)
ax.set_yticks(range(len(fm))[::-1], fm.label)
ax.set_xlim(-0.035, 0.125)
ax.set_xlabel('Mean ΔPLV (PLV − null median)')
ax.set_title('Coupling by region set: fasted → fed', pad=24, loc='left')
ax.text(0.0, 1.02, '● fasted', transform=ax.transAxes, fontsize=11.5, color=FA)
ax.text(0.2, 1.02, '● fed', transform=ax.transAxes, fontsize=11.5, color=FE)
ax.text(0.34, 1.02, 'raw means · right: adjusted fed − fasted', transform=ax.transAxes, fontsize=11,
        color=SUB)
style(ax, grid='x')
letter(ax, 'A', x=-0.36)

# parcel-wise effect size (Cohen's dz of fed - fasted subject-mean dPLV), on axial template slices
par = pd.read_csv(os.path.join(RES, 'synchrony', 'parcel_runs.csv'))
par = par[par.in_primary_analysis]
sm = par.pivot_table(index=['region', 'subject_id'], columns='state', values='dplv', aggfunc='mean').dropna()
dd = (sm.fed - sm.fasted).groupby('region')
dz = (dd.mean() / dd.std()).rename(lambda r: int(r[1:]))
pimg = nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz'))
pv = np.asanyarray(pimg.dataobj).astype(int)
dzvol = np.zeros(pv.shape, dtype=float)
for k, v in dz.items():
    dzvol[pv == k] = v
dz_img = nib.Nifti1Image(dzvol, pimg.affine)
reb_img = None if SEND else nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz'))

views4 = [('lh', 'lateral', 'Left lateral'), ('rh', 'lateral', 'Right lateral'),
          ('lh', 'medial', 'Left medial'), ('rh', 'medial', 'Right medial')]
for j, (h, vw, vlab) in enumerate(views4):
    axs = fig.add_axes([0.47 + (j % 2) * 0.235, 0.43 - (j // 2) * 0.33, 0.23, 0.33], projection='3d')
    norm4, cm4 = surface_panel(axs, dz_img, h, vw, 'RdBu_r', -0.8, 0.8, outline_img=reb_img)
    axs.set_title(vlab, fontsize=12.5, color=SUB, pad=-2)
cax = fig.add_axes([0.955, 0.2, 0.012, 0.45])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm4, cmap=cm4), cax=cax, orientation='vertical')
cb.set_label("Fed − fasted ΔPLV, Cohen's dz", fontsize=12.5)
cb.ax.tick_params(labelsize=11)
fig.text(0.47, 0.905, 'B', fontsize=17, fontweight='bold', color=INK)
fig.text(0.5, 0.905, 'Regional effect of feeding (cortex)', fontsize=14, fontweight='bold', color=INK)
fig.text(0.5, 0.87, f'{int((dz > 0).sum())} of {len(dz)} regions higher after feeding · none significant after FDR'
         + ('' if SEND else ' · black outline: Rebollo 2018 gastric-coupled regions'), fontsize=11.5, color=SUB)
fig.suptitle('Network localization', fontsize=15, fontweight='bold', color=INK, x=0.05, ha='left', y=0.98)
save(fig, 'fig4_localization', 'Figure 4',
     'Network localization. (A) Mean ΔPLV per region set at the fasted (blue) and fed (orange) visits; numbers give the '
     'covariate-adjusted fed minus fasted estimate, its p value and the FDR-corrected q across the five region sets. '
     "(B) Fed minus fasted effect size (Cohen's dz of ΔPLV) for each of the 200 cortical regions in the 15 participants "
     'with usable EGG in both states, on the inflated fsaverage surface (red, higher after feeding; blue, lower). The '
     'map is unthresholded and descriptive: no region survived FDR correction (smallest q = 0.50).'
     + ('' if SEND else ' The black outline marks the regions coupled to the gastric rhythm in Rebollo et al. (2018).'))


# ================================================================ Figure 5: cortical surface maps
par_all = pd.read_csv(os.path.join(RES, 'synchrony', 'parcel_runs.csv'))
pp = par_all[par_all.in_primary_analysis]
smp = pp.pivot_table(index=['region', 'subject_id'], columns='state', values='dplv', aggfunc='mean').dropna()
dfd = (smp.fed - smp.fasted).groupby('region')
dz_par = (dfd.mean() / dfd.std()).rename(lambda r: int(r[1:]))
pimg = nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz'))
pvol = np.asanyarray(pimg.dataobj).astype(int)
reb_img = None if SEND else nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz'))


def parcel_img(values):
    vol = np.zeros(pvol.shape)
    for k, v in values.items():
        vol[pvol == k] = v
    return nib.Nifti1Image(vol, pimg.affine)


from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
SET_COLORS = ['#1baf7a', '#2a78d6', '#eda100', '#e87ba4', '#8a8984']   # insula, SMN, salience, DMN, subcortical
fam_img = nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz'))
rows = [('Region sets\n(analysis map)', fam_img, ListedColormap(SET_COLORS), (0.5, 5.5)),
        ('Fed − fasted\n(effect size dz)', parcel_img(dz_par), 'PuOr_r', (-0.8, 0.8))]
if SEND:
    rows = rows[:1]                               # the effect map is already shown in Figure 4B
views = [('lh', 'lateral', 'Left lateral'), ('lh', 'medial', 'Left medial'),
         ('rh', 'medial', 'Right medial'), ('rh', 'lateral', 'Right lateral')]
if SEND:
    # A: the five atlas region sets; B: the eight CONN ICA networks (both are used in the network tables)
    cimg = nib.load(os.path.join(BASE, 'atlas', 'conn_networks_3mm.nii.gz'))
    cmask = np.asanyarray(cimg.dataobj) > 0
    cnames = [l.split('.')[0] for l in open(os.path.join(BASE, 'atlas', 'conn_networks_labels.txt')) if l.strip()]
    CONN_ORDER = ['DefaultMode', 'SensoriMotor', 'Visual', 'Salience', 'DorsalAttention', 'FrontoParietal', 'Language',
                  'Cerebellar']
    CONN_LAB = ['Default mode', 'Sensorimotor', 'Visual', 'Salience', 'Dorsal attention', 'Frontoparietal', 'Language',
                'Cerebellar (not on surface)']
    CONN_COLORS = ['#e87ba4', '#2a78d6', '#59a14f', '#eda100', '#76b7b2', '#b07aa1', '#9c755f', '#8a8984']
    clab = np.zeros(cmask.shape[:3])
    for j, nm in enumerate(cnames):
        clab[cmask[..., j] & (clab == 0)] = CONN_ORDER.index(nm) + 1
    conn_img = nib.Nifti1Image(clab, cimg.affine)
    panels = [('A', 'Atlas region sets: three functional networks (Yeo), the insula and the subcortical nuclei',
               rows[0][1], ListedColormap(SET_COLORS), (0.5, 5.5),
               ['Insula (anatomical)', 'Sensorimotor network', 'Salience network', 'Default-mode network'], SET_COLORS[:4]),
              ('B', 'CONN ICA-derived networks (32 masks grouped into eight networks)', conn_img,
               ListedColormap(CONN_COLORS), (0.5, 8.5), CONN_LAB[:7], CONN_COLORS[:7])]   # cerebellum: not on the surface
    H = 13.6                                      # figure height (inches); rows A and B as before, C below
    fig = plt.figure(figsize=(15, H))
    for i, (letter_, ptitle, img, cmap, lim, names_, cols_) in enumerate(panels):
        b = H - 4.28 - i * 4.42                   # bottom of row i (inches)
        fig.text(0.015, (b + 3.62) / H, letter_, fontsize=17, fontweight='bold', color=INK)
        fig.text(0.04, (b + 3.62) / H, ptitle, fontsize=13.5, fontweight='bold', color=INK)
        for j, (h, vw, vlab) in enumerate(views):
            ax = fig.add_axes([0.025 + j * 0.24, (b + 0.56) / H, 0.23, 2.73 / H], projection='3d')
            surface_panel(ax, img, h, vw, cmap, lim[0], lim[1], outline_img=None)
            if i == 0:
                fig.text(0.14 + j * 0.24, (b + 3.29) / H, vlab, ha='center', fontsize=12.5, color=SUB, fontweight='bold')
        fig.legend([Patch(color=c) for c in cols_], names_, loc='lower center', ncol=4, frameon=False,
                   fontsize=12, bbox_to_anchor=(0.5, (b - 0.33) / H))
    # C: the subcortical nuclei (Tian S1, labels 201-216 in the parcel image) on MNI slices
    from scipy.ndimage import map_coordinates
    tmpl = nib.load(TEMPLATE)
    tvol = np.asanyarray(tmpl.dataobj).astype(float).reshape(tmpl.shape[:3])
    NUC = ['Hippocampus', 'Amygdala', 'Posterior thalamus', 'Anterior thalamus', 'Nucleus accumbens',
           'Globus pallidus', 'Putamen', 'Caudate']                  # Tian S1 order, right then left
    NUC_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']
    nuc = np.where(pvol > 200, (pvol - 201) % 8 + 1, 0)

    def mni_slice(axis, at, r1, r2):
        u, v = np.meshgrid(np.arange(*r1, 0.5), np.arange(*r2, 0.5))   # in-plane MNI coordinates (mm)
        xyz = [u, v, np.full(u.shape, float(at))] if axis == 'z' else [u, np.full(u.shape, float(at)), v]
        w = np.stack([*xyz, np.ones(u.shape)]).reshape(4, -1)
        t = map_coordinates(tvol, (np.linalg.inv(tmpl.affine) @ w)[:3], order=1).reshape(u.shape)
        lab = map_coordinates(nuc, np.round(np.linalg.inv(pimg.affine) @ w)[:3], order=0).reshape(u.shape)
        return t, lab

    b = H - 4.28 - 2 * 4.42
    fig.text(0.015, (b + 3.62) / H, 'C', fontsize=17, fontweight='bold', color=INK)
    fig.text(0.04, (b + 3.62) / H, 'Subcortical nuclei (Tian scale I, 16 regions), shown on MNI slices',
             fontsize=13.5, fontweight='bold', color=INK)
    SLICES = [('z', -18, 'Axial, z = −18 mm'), ('z', 0, 'Axial, z = 0 mm'), ('y', 10, 'Coronal, y = +10 mm'),
              ('y', -4, 'Coronal, y = −4 mm')]          # together the four slices cross all eight nuclei
    for j, (axis, at, slab) in enumerate(SLICES):
        ax = fig.add_axes([0.035 + j * 0.24, (b + 0.5) / H, 0.21, 2.75 / H])
        r2 = (-48, 34) if axis == 'z' else (-38, 44)
        t, lab = mni_slice(axis, at, (-48, 48), r2)
        ext = (-48, 48, r2[0], r2[1])
        ax.imshow(t, cmap='gray', origin='lower', extent=ext, vmin=np.percentile(t, 2), vmax=np.percentile(t, 99.5))
        ax.imshow(np.ma.masked_equal(lab, 0), cmap=ListedColormap(NUC_COLORS), vmin=0.5, vmax=8.5, origin='lower',
                  extent=ext, alpha=0.9, interpolation='nearest')
        for xx, side, ha in [(-45, 'L', 'left'), (45, 'R', 'right')]:    # +x is the right hemisphere
            ax.text(xx, r2[1] - 3, side, color='white', fontsize=12, fontweight='bold', va='top', ha=ha)
        ax.set_title(slab, fontsize=12.5, color=SUB, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.legend([Patch(color=c) for c in NUC_COLORS], NUC, loc='lower center', ncol=4, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, (b - 0.42) / H))
    fig.suptitle('Regions and networks used in the analysis', fontsize=17, fontweight='bold', color=INK, x=0.02,
                 ha='left', y=0.995)
    rows = []
else:
    fig = plt.figure(figsize=(15, 7.6))
for i, (rlab, img, cmap, lim) in enumerate(rows):
    for j, (h, vw, vlab) in enumerate(views):
        ax = fig.add_axes([0.16 + j * 0.2, 0.53 - i * 0.43, 0.195, 0.36], projection='3d')
        norm, cm = surface_panel(ax, img, h, vw, cmap, lim[0], lim[1], outline_img=reb_img)
        if i == 0:
            ax.set_title(vlab, fontsize=13, color=SUB, pad=-4)
    fig.text(0.075, 0.71 - i * 0.43, rlab, ha='center', va='center', fontsize=14, fontweight='bold', color=INK)
    fig.text(0.015, 0.89 - i * 0.43, 'AB'[i], fontsize=17, fontweight='bold', color=INK)
    if i == 0:
        names_ = ['Insula', 'Sensorimotor', 'Salience', 'Default mode', 'Subcortical (not on surface)']
        fig.legend([Patch(color=c) for c in SET_COLORS] + ([] if SEND else [Patch(facecolor='white', edgecolor='#111111')]),
                   names_ + ([] if SEND else ['Rebollo 2018 regions (outline)']), loc='upper center', ncol=6, frameon=False,
                   fontsize=12, bbox_to_anchor=(0.56, 0.955))
    else:
        cax = fig.add_axes([0.965, 0.14, 0.011, 0.28])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax)
        cb.ax.tick_params(labelsize=11)
if not SEND:
    fig.suptitle('Cortical maps of stomach–brain coupling', fontsize=17, fontweight='bold', color=INK, x=0.02,
                 ha='left', y=0.99)
save(fig, 'fig5_surface', 'Figure 5', 'Regions and networks used in the analysis. (A) The five atlas region sets on '
     'the inflated fsaverage surface. They mix two kinds of region: three functional networks (sensorimotor, salience '
     'and default mode; Schaefer-200 parcels grouped by their Yeo-7 network labels) and two anatomical structures, the '
     'insula (Harvard-Oxford) and the subcortical nuclei (Tian scale I), which lie below the cortical surface and are '
     'shown in C. (B) The eight CONN networks, derived from group independent component analysis and defined by 32 '
     'masks; the cerebellar network is not visible on the cortical surface. Networks with the same name in A and B are '
     'defined differently and are analysed separately. (C) The 16 subcortical regions (eight nuclei in each hemisphere) '
     'on axial and coronal slices of the MNI 2009c template, at the 3-mm resolution used in the analysis; L and R mark '
     'the left and right hemispheres.') if SEND else save(fig, 'fig5_surface', 'Figure 5',
     'Cortical surface maps (fsaverage, inflated). (A) The predefined region sets used in the analysis: insula '
     '(Harvard-Oxford), sensorimotor, salience and default-mode networks (Schaefer-200 parcels with Yeo-7 labels); '
     'the subcortical nuclei (Tian S1) are not visible on the cortical surface. (B) Fed minus fasted difference in '
     'ΔPLV for each of the 200 cortical regions in the 15 participants with usable EGG in both states, shown as '
     "Cohen's dz (orange, higher after feeding; purple, lower). The map is unthresholded and descriptive: no region "
     'survived FDR correction (smallest q = 0.50). In both rows the black outline marks the regions coupled to the gastric rhythm in '
     'Rebollo et al. (2018). MNI-space values were sampled between the white and pial surfaces.')
print('figure 5 written')
