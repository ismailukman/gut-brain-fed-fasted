"""
Abstract figures.

fig_results   Figure 2: (A) blood glucose and (B) the fraction of grey-matter voxels with gastric synchronization,
              fasted vs fed, in the participants with usable EGG at both visits
example run   the signals of one fed run (243929_13B_fe run 2, chosen for illustration) are cached in
              results/synchrony/abstract_example.npz for Figure 1 (13_figure1_methods.py); computing the cache needs the
              BOLD data and the raw EGG, later runs read the cache

Output: figures/abstract/fig_results{,_nocap}.{png,pdf}, captions.json
Usage: python 12_figures_abstract.py
"""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import signal, stats

from gutsync import TR, analytic, bandpass, egg_peak, load_bold, load_egg, store_caption
from config import BASE
from config import RAW as ROOT

RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, 'figures', 'abstract')
CACHE = os.path.join(RES, 'synchrony', 'abstract_example.npz')
EX_ID, EX_RUN = '243929_13B_fe', 2
os.makedirs(OUT, exist_ok=True)

FA, FE = '#2a78d6', '#eb6834'                      # fasted, fed
INK, SUB, MUTED, GRID = '#1f1f1e', '#55554f', '#9a9993', '#ebeae6'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': SUB, 'ytick.color': SUB, 'axes.titleweight': 'bold',
                     'axes.titlesize': 13, 'axes.titlecolor': INK, 'legend.fontsize': 11})


def style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def letter(ax, s, x=-0.12, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=16, fontweight='bold', color=INK, va='bottom')


def save(fig, name, label, caption):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}_nocap.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
    store_caption(OUT, name, label, caption)
    cap = fig.text(0.0, -0.01, '\n'.join(textwrap.wrap(f'{label}. {caption}', 210)), ha='left', va='top',
                   fontsize=10.5, color=SUB)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
    cap.remove()
    plt.close(fig)


# ================================================================ example run (cached)
def plv_aw(z, zg):
    a, ag = np.abs(z), np.abs(zg)
    u, ug = z / np.where(a > 0, a, 1), zg / ag
    return np.abs(u.T @ np.conj(ug)) / len(z), np.abs(z.T @ np.conj(zg)) / np.maximum(a.T @ ag, 1e-30)



def cache_example():
    if not os.path.exists(CACHE):
        import bioread
        man = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
        man = man[man.egg_pass].reset_index(drop=True)
        ex = man[(man.imaging_id == EX_ID) & (man.run == EX_RUN)].iloc[0]
        egg_dir = os.path.dirname(os.path.join(ROOT, ex.egg_filtered_file))
        dom = int(np.load(os.path.join(egg_dir, f'dominant_channel{EX_ID}_run{EX_RUN}strict.npy')))
        f0 = egg_peak(os.path.join(ROOT, ex.egg_filtered_file))
        acq = bioread.read_file(os.path.join(ROOT, 'Multi_sub_GastricfMRI', 'egg_data', f'sub-{EX_ID}_egg_run{EX_RUN}.acq'))
        raw = np.array([signal.resample_poly(c.data[:int(900 * c.samples_per_second)], 1,
                                             int(c.samples_per_second / 10)) for c in acq.channels[:3]])
        fpsd, psd = signal.welch(raw, fs=10, nperseg=2000, noverlap=1000)    # 200-s windows, 100-s overlap

        family = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'families_3mm.nii.gz')).dataobj).astype(int)
        pimg = nib.load(os.path.join(BASE, 'atlas', 'parcels216_3mm.nii.gz'))
        parcels = np.asanyarray(pimg.dataobj).astype(int)
        rebollo = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'rebollo2018_3mm.nii.gz')).dataobj) > 0
        conn = np.asanyarray(nib.load(os.path.join(BASE, 'atlas', 'conn_networks_3mm.nii.gz')).dataobj) > 0
        gm_idx = np.flatnonzero(((parcels > 0) | rebollo | conn.any(-1)).ravel())
        egg = {(r.imaging_id, r.run): load_egg(os.path.join(ROOT, r.egg_filtered_file)) for r in man.itertuples()}
        n_t = len(egg[(EX_ID, EX_RUN)])
        y, _ = load_bold(ex.bold_file, gm_idx)
        y = y[:n_t]
        valid = y.std(0) > 0
        zb = analytic(bandpass(y - y.mean(0), f0, 1 / TR))
        zg = analytic(bandpass(egg[(EX_ID, EX_RUN)], f0, 1 / TR))
        keys = [k for k in egg if k[0][:9] != EX_ID[:9] and len(egg[k]) >= n_t]
        pool = np.column_stack([analytic(bandpass(egg[k][:n_t], f0, 1 / TR)) for k in keys])
        p_own, a_own = plv_aw(zb, zg[:, None])
        p_nul, a_nul = plv_aw(zb, pool)
        p_own, a_own = p_own[:, 0], a_own[:, 0]
        p95 = np.percentile(p_nul, 95, axis=1)
        above = valid & (p_own > p95)
        rank = (p_nul < p_own[:, None]).mean(1)
        # illustrative voxel: a typical above-chance sensorimotor voxel (75th percentile of own-EGG PLV among them)
        cand = np.flatnonzero(above & (family.ravel()[gm_idx] == 2))
        v = cand[np.argmin(np.abs(p_own[cand] - np.percentile(p_own[cand], 75)))]
        m = int(np.argmin(np.abs(p_nul[v] - np.median(p_nul[v]))))              # a typical mismatched EGG
        vol = lambda x: np.bincount(gm_idx, weights=x.astype(float), minlength=family.size).reshape(family.shape)
        np.savez(CACHE, raw=raw, dom=dom, f0=f0, fpsd=fpsd, psd=psd, zg=zg, zb_v=zb[:, v], zm_v=pool[:, m],
                 y_v=y[:, v] - y[:, v].mean(), p_own_v=p_own[v], a_own_v=a_own[v], p_nul_v=p_nul[v], a_nul_v=a_nul[v],
                 above=vol(above).astype(np.int8), valid=vol(valid).astype(np.int8),
                 vox=np.array(np.unravel_index(gm_idx[v], family.shape)), affine=pimg.affine,
                 prom=above[valid].mean(), rank_v=rank[v], n_null=len(keys))


cache_example()

subj = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_subjects.csv'))
fam = pd.read_csv(os.path.join(RES, 'synchrony', 'family_runs.csv'))
fam = fam[fam.in_primary_analysis & (fam.region == 'all-grey')]
prom = fam.pivot_table(index='subject_id', columns='state', values='frac_vox_above_null') * 100
tf = pd.read_csv(os.path.join(RES, 'synchrony', 'threshold_free_runs.csv'))
tf = tf[tf.in_primary_analysis & (tf.region == 'all-grey')].pivot_table(index='subject_id', columns='state',
                                                                          values='tf_plv')
mods = pd.read_csv(os.path.join(RES, 'aim2_models.csv')).set_index(['analysis', 'set'])
PM = mods.loc[('Prominence: grey-matter fraction of gastric-coupled voxels', 'paired, FD covariate')]
TFM = pd.read_csv(os.path.join(RES, 'threshold_free_prominence.csv'))
TFM = TFM[(TFM.region == 'all-grey') & (TFM.metric == 'PLV') & (TFM.measure == 'threshold-free score')].iloc[0]
g = subj.set_index('subject_id')[['glucose_fasted_mg_dl', 'glucose_fed_mg_dl']].dropna()
g.columns = ['fasted', 'fed']
g = g[g.index.isin(prom.index)]          # the same participants as panel B (paired set) with glucose at both visits
pg = stats.wilcoxon(g.fed, g.fasted).pvalue


def fmt_p(p):
    return 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'


def raincloud(ax, sm, ylab, title, sub, ref=None, ref_lab=None):
    for x, col, cc in ((0, 'fasted', FA), (1, 'fed', FE)):
        v = sm[col].values
        kde = stats.gaussian_kde(v)
        yy = np.linspace(v.min() - 0.25 * v.std(), v.max() + 0.25 * v.std(), 200)
        d = kde(yy) / kde(yy).max() * 0.32
        side = -1 if x == 0 else 1
        ax.fill_betweenx(yy, x + side * 0.12, x + side * (0.12 + d), color=cc, alpha=0.28, lw=0)
        ax.plot(x + side * (0.12 + d), yy, color=cc, lw=1.2)
    for _, r in sm.iterrows():
        ax.plot([0, 1], [r.fasted, r.fed], color=FE if r.fed > r.fasted else (FA if r.fed < r.fasted else MUTED),
                lw=1.1, alpha=0.35 if r.fed != r.fasted else 0.8, zorder=1)
    ax.scatter(np.zeros(len(sm)), sm.fasted, s=40, color=FA, ec='white', lw=1, zorder=3)
    ax.scatter(np.ones(len(sm)), sm.fed, s=40, color=FE, ec='white', lw=1, zorder=3)
    for x, col in ((0, 'fasted'), (1, 'fed')):
        mu, se = sm[col].mean(), sm[col].std() / np.sqrt(len(sm))
        ax.errorbar(x, mu, yerr=1.96 * se, fmt='o', color=INK, ms=7, mfc='white', mew=2, elinewidth=2, capsize=0,
                    zorder=4)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=0)
        ax.text(1.98, ref, ref_lab, fontsize=13, color=INK, va='bottom', ha='right')
    ax.set_xticks([0, 1], ['Fasted', 'Fed'])
    for t_, cc in zip(ax.get_xticklabels(), (FA, FE)):
        t_.set_color(cc); t_.set_fontweight('bold')
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylabel(ylab)
    ax.set_title(title, pad=26)
    ax.text(0.5, 1.025, sub, transform=ax.transAxes, ha='center', fontsize=13.5, color=INK)
    style(ax)



plt.rcParams.update({'font.size': 14.5, 'axes.labelsize': 15, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.titlesize': 16, 'legend.fontsize': 13.5})
# ---------------------------------------------------------------- abstract version: panels A and B only
_mm = pd.read_csv(os.path.join(BASE, 'data', 'gutbff_analysis_manifest.csv'))
_nr = _mm[_mm.in_primary_analysis].groupby('state').size()
fig = plt.figure(figsize=(14, 6.6))
gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.8, bottom=0.1, wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
raincloud(ax, g, 'Blood glucose (mg/dL)', 'Blood glucose',
          f'+{(g.fed - g.fasted).mean():.1f} mg/dL · {fmt_p(pg)}')
letter(ax, 'A', x=-0.24)
ax = fig.add_subplot(gs[0, 1])
raincloud(ax, prom, 'Brain voxels (grey matter) with\ngastric synchronization (%)', 'Brain voxels with gastric synchronization',
          f'+{100 * PM.estimate_fed_minus_fasted:.1f} points · {fmt_p(PM.p_two_sided)} · '
          f'{int((prom.fed > prom.fasted).sum())} of {len(prom)} higher', ref=6.5, ref_lab='chance\n(6.5%)')
letter(ax, 'B', x=-0.24)
fig.text(0.08, 0.95, f'Breakfast, blood glucose and stomach-brain synchronization ({len(prom)} participants with usable '
         'EGG at both visits)', fontsize=15, fontweight='bold', color=INK)
save(fig, 'fig_results', 'Figure 2',
     'Fasted and fed visits in the {n} participants with usable EGG at both visits ({fa} fasted and {fe} fed runs). (A) Blood glucose in the {ng} with values at both visits (Wilcoxon test). (B) Fraction of grey-matter voxels with gastric synchronization; dashed line, chance (about 6.5%). Change and P from a linear mixed model using all runs. Points, participant means; lines join each participant (orange, higher when fed; blue, lower; grey, unchanged); white circles, mean and 95% CI.'.format(n=len(prom), fa=_nr['fasted'], fe=_nr['fed'], ng=len(g)))
print('done')
