"""
Figure 1 of the abstract: study design, analysis pipeline and how the fraction of synchronized voxels is measured.

  A  within-participant design (fasted and fed visits, counterbalanced)
  B  pipeline for one run: raw EGG -> gastric peak -> band-pass; preprocessed BOLD of one voxel -> same band-pass;
     Hilbert phase -> PLV in every grey-matter voxel
  C  chance and prominence: the voxel's PLV against the mismatched-participant chance distribution, and the share
     of grey-matter voxels above chance in the run

The small panels are drawn from one fed run (243929_13B_fe run 2, chosen for illustration) using the cache written by
12_figures_abstract.py (results/synchrony/abstract_example.npz) and placed in a TikZ layout compiled with tectonic
(pdftoppm converts it to PNG).
Output: figures/abstract/fig_methods{,_nocap}.{pdf,png}; caption in figures/abstract/captions.json

Usage: python 13_figure1_methods.py
"""
import os
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import signal
from scipy.ndimage import map_coordinates

from gutsync import TR, TRIM, store_caption

from config import BASE, TEMPLATE
CACHE = os.path.join(BASE, 'results', 'synchrony', 'abstract_example.npz')
OUT = os.path.join(BASE, 'figures', 'abstract')
WORK = os.path.join(OUT, 'fig1_parts')
os.makedirs(WORK, exist_ok=True)

FA, FE = '#2a78d6', '#eb6834'
INK, SUB, MUTED, GRID = '#1f1f1e', '#55554f', '#9a9993', '#ebeae6'
BAND = '#008b82'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                     'axes.labelsize': 9, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})

c = np.load(CACHE)
f0, dom = float(c['f0']), int(c['dom'])
zg, zb = c['zg'], c['zb_v']
aff = c['affine']
t_tr = (np.arange(len(zg)) + TRIM) * TR
win = (t_tr >= 300) & (t_tr <= 600)
tmpl = nib.load(TEMPLATE)
tvol = np.asanyarray(tmpl.dataobj).astype(float).reshape(tmpl.shape[:3])


def part(name, w, h):
    fig, ax = plt.subplots(figsize=(w / 2.54, h / 2.54))
    return fig, ax


def done(fig, name):
    fig.savefig(os.path.join(WORK, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close(fig)


def bare(ax, xlabel=None):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=1)
    ax.tick_params(length=2, pad=1)


def mni_slice(vol3, axis, at, r1=(-72, 72), r2=(-104, 72)):
    u, w = np.meshgrid(np.arange(*r1, 1.0), np.arange(*r2, 1.0))
    xyz = [u, w, np.full(u.shape, float(at))] if axis == 'z' else [np.full(u.shape, float(at)), u, w]
    hh = np.stack([*xyz, np.ones(u.shape)]).reshape(4, -1)
    t = map_coordinates(tvol, (np.linalg.inv(tmpl.affine) @ hh)[:3], order=1).reshape(u.shape)
    lab = map_coordinates(vol3.astype(float), np.round(np.linalg.inv(aff) @ hh)[:3], order=0).reshape(u.shape)
    return t, lab, (r1[0], r1[1], r2[0], r2[1])


def show_slice(ax, vol3, at, color):
    t, lab, ext = mni_slice(vol3, 'z', at)
    brain = t > np.percentile(t, 35)
    ax.imshow(np.where(brain, t, np.nan), cmap='gray', origin='lower', extent=ext,
              vmin=np.percentile(t[brain], 2), vmax=np.percentile(t[brain], 99.5))
    ax.imshow(np.ma.masked_less(lab, 0.5), cmap=ListedColormap([color]), origin='lower', extent=ext, alpha=0.95,
              interpolation='nearest')
    ax.axis('off')


# ---------------------------------------------------------------- B: EGG branch
raw = c['raw']
tt = np.arange(raw.shape[1]) / 10
w = (tt >= 300) & (tt <= 600)
fig, ax = part('egg_raw', 4.2, 2.3)
for k in range(3):
    x = signal.detrend(raw[k][w])
    x = np.clip(x / (5 * x.std()), -0.6, 0.6)
    ax.plot(tt[w], x - 1.9 * k, color=INK if k == dom else MUTED, lw=0.6 if k == dom else 0.45)
    ax.text(302, -1.9 * k + 0.68, f'channel {k + 1}', fontsize=7.8, color=INK if k == dom else SUB,
            fontweight='bold' if k == dom else 'normal')
ax.set_xlim(300, 600)
ax.set_ylim(-4.5, 1.05)
bare(ax, 'Time (s)')
done(fig, 'egg_raw')

fpsd, psd = c['fpsd'], c['psd']
cpm = fpsd * 60
sel = (cpm >= 1) & (cpm <= 8)
fig, ax = part('egg_psd', 4.2, 2.3)
ax.axvspan(2, 4, color=GRID, lw=0, zorder=0)
ax.axvspan((f0 - 0.015) * 60, (f0 + 0.015) * 60, color=BAND, alpha=0.2, lw=0, zorder=1)
top = psd[dom][sel].max()
for k in range(3):
    ax.plot(cpm[sel], psd[k][sel] / top, color=INK if k == dom else MUTED, lw=1.2 if k == dom else 0.6,
            zorder=3 if k == dom else 2)
ax.axvline(f0 * 60, color=BAND, lw=1, ls='--', zorder=2)
ax.text(f0 * 60 + 0.35, 0.92, 'f$_0$ ≈ 3 cpm', fontsize=9.5, color=BAND, fontweight='bold', va='top')
ax.set_xlim(1, 8)
ax.set_ylim(0, 1.12)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('Frequency (cpm)', labelpad=1)
ax.set_xticks([2, 4, 6, 8])
ax.set_ylabel('Power', labelpad=1)
ax.set_yticks([0, 1])
ax.tick_params(length=2, pad=1)
done(fig, 'egg_psd')


def trace(name, z, col, raw_v=None):
    fig, ax = part(name, 4.2, 2.3)
    zz = z / np.abs(z)[win].std() / 3
    if raw_v is not None:
        ax.plot(t_tr[win], raw_v[win] / raw_v[win].std() / 3, color=col, lw=0.5, alpha=0.3)
    ax.plot(t_tr[win], zz.real[win], color=col, lw=1.0)
    ax.set_xlim(300, 600)
    bare(ax, 'Time (s)')
    done(fig, name)


trace('egg_filt', zg, INK)
trace('bold_filt', zb, FE, raw_v=c['y_v'][TRIM:-TRIM])

# BOLD voxel location
fig, ax = part('bold_vox', 2.3, 2.3)
vx = c['vox']
xyz = aff @ np.r_[vx, 1]
mark = np.zeros(c['valid'].shape, np.int8)
mark[tuple(vx)] = 1
show_slice(ax, mark, xyz[2], FE)
ax.plot(xyz[0], xyz[1], 'o', ms=9, mfc='none', mec=FE, mew=1.5)
done(fig, 'bold_vox')

# Hilbert phases
pe, pb = np.angle(zg), np.angle(zb)
dphi = np.angle(zb * np.conj(zg))
mu = np.angle(np.mean(np.exp(1j * dphi)))
fig, ax = part('phase', 4.6, 3.6)
gap = 3.1 * np.pi
for p_, off, col in ((pe, 0, INK), (pb, -gap, FE)):
    pp = p_[win].copy()
    pp[np.r_[np.abs(np.diff(pp)) > np.pi, False]] = np.nan
    ax.plot(t_tr[win], pp + off, color=col, lw=0.9)
ax.scatter(t_tr[win], np.angle(np.exp(1j * (dphi[win] - mu))) - 2 * gap, s=1.5, color=BAND)
for yy, lab, col in ((0, 'EGG phase φ$_E$', INK), (-gap, 'BOLD phase φ$_B$', FE), (-2 * gap, 'difference Δφ', BAND)):
    ax.text(302, yy + np.pi + 0.3, lab, fontsize=8.1, color=col, fontweight='bold')
ax.set_xlim(300, 600)
ax.set_ylim(-2 * gap - np.pi - 0.3, np.pi + 1.6)
bare(ax, 'Time (s)')
done(fig, 'phase')

# ---------------------------------------------------------------- C: chance and prominence
nv = c['p_nul_v']
own = float(c['p_own_v'])
p95 = np.percentile(nv, 95)
fig, ax = part('chance', 5.2, 3.0)
jit = (np.random.default_rng(3).random(len(nv)) - 0.5) * 0.5
ax.scatter(nv, jit, s=9, color=MUTED, alpha=0.85, lw=0, zorder=2)
ax.plot([p95, p95], [-0.45, 0.3], color=INK, lw=0.9, ls='--', zorder=1)
ax.text(p95, -0.5, '95th\npercentile', fontsize=8.1, color=INK, ha='center', va='top')
ax.scatter([own], [0], s=60, marker='D', color=FE, ec='white', lw=0.8, zorder=4)
ax.text(own + 0.04, 0, 'own EGG', fontsize=8.1, color=FE, fontweight='bold', ha='left', va='center')
ax.text(nv.min(), 0.36, 'other participants\' EGG (chance)', fontsize=8.1, color=INK, ha='left', va='bottom')
ax.set_ylim(-0.95, 0.75)
ax.set_xlim(0, max(own, nv.max()) * 1.5)
bare(ax, 'PLV at this voxel')
done(fig, 'chance')

fig, axs = plt.subplots(1, 3, figsize=(6.6 / 2.54, 2.6 / 2.54), gridspec_kw={'wspace': 0.03})
for a_, zc in zip(axs, (-8, 22, 52)):
    show_slice(a_, c['above'], zc, FE)
    a_.text(0.5, -0.04, f'z = {zc:+d}'.replace('-', '−'), transform=a_.transAxes, ha='center', va='top', fontsize=7.8,
            color=INK)
done(fig, 'prom')

# ---------------------------------------------------------------- TikZ layout
PROM = 100 * float(c['prom'])
NNULL = len(nv)
TEX = r'''\documentclass{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{navy}{HTML}{073B74}
\definecolor{ink}{HTML}{1F1F1E}
\definecolor{sub}{HTML}{55554F}
\definecolor{fasted}{HTML}{2A78D6}
\definecolor{fed}{HTML}{EB6834}
\definecolor{band}{HTML}{008B82}
\definecolor{paleBlue}{HTML}{EDF7FF}
\definecolor{paleOrange}{HTML}{FFF3E9}
\definecolor{paleTeal}{HTML}{E8F6F5}
\tikzset{
 panel/.style={rounded corners=3pt,draw=black,line width=.5pt,fill=white},
 badge/.style={circle,fill=black,text=white,font=\bfseries\large,minimum size=7.5mm,inner sep=0pt},
 ptitle/.style={font=\bfseries\fontsize{11}{13}\selectfont,text=navy,anchor=west},
 capt/.style={font=\fontsize{8.4}{10}\selectfont,text=ink,align=center},
 lab/.style={font=\bfseries\fontsize{9}{10.5}\selectfont,align=center},
 flow/.style={-{Stealth[length=3.4mm,width=2.6mm]},line width=1.6pt,draw=navy!80,shorten <=2pt,shorten >=2pt},
 eqbox/.style={rounded corners=2pt,draw=#1!70,fill=#1!8,line width=.6pt,inner sep=4pt,align=center},
}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm]
% ------------------------------------------------ title
\node[font=\bfseries\fontsize{14}{16}\selectfont,text=navy,anchor=west] at (0,21.0)
  {Stomach--brain synchronization: design and analysis};

% ------------------------------------------------ A design
\draw[panel] (0,16.75) rectangle (19.6,20.45);
\node[badge] at (.55,19.95) {A}; \node[ptitle] at (1.15,19.95) {Within-participant design};
\node[capt,anchor=west,align=left] at (1.15,19.42) {25 healthy adults; two morning visits 1--14 days apart in counterbalanced order (14 fasted first, 11 fed first);\\3T Philips Ingenia Elition, TR = 2 s; 3-channel cutaneous EGG recorded during each scan};
\foreach \x/\col/\pale/\name/\meal in {.35/fasted/paleBlue/Fasted visit/overnight fast, 10.05/fed/paleOrange/Fed visit/usual breakfast}{
  \draw[rounded corners=4pt,draw=\col,fill=\pale,line width=.9pt] (\x,16.98) rectangle ++(9.2,1.85);
  \node[font=\bfseries\fontsize{10}{12}\selectfont,text=\col,anchor=west] at (\x+.2,18.25) {\name};
  \node[font=\fontsize{8.5}{10}\selectfont,text=ink,anchor=west] at (\x+.2,17.75) {\meal};
  \foreach \dx/\w/\txt in {2.75/1.45/{Blood\\glucose}, 4.4/2.25/{rsfMRI + EGG\\run 1, 15 min}, 6.8/2.25/{rsfMRI + EGG\\run 2, 15 min}}{
    \node[draw=\col,fill=white,rounded corners=2pt,line width=.6pt,minimum width=\w cm,minimum height=1.15cm,
          font=\fontsize{8}{9.5}\selectfont,align=center,text=ink] at (\x+\dx+\w/2,17.9) {\txt};}
}
\draw[{Stealth[length=2mm]}-{Stealth[length=2mm]},line width=.8pt,draw=sub] (9.6,17.9) -- (10.0,17.9);

% ------------------------------------------------ B pipeline
\draw[panel] (0,6.35) rectangle (19.6,16.5);
\node[badge] at (.55,16.0) {B}; \node[ptitle] at (1.15,16.0) {From EGG and BOLD to phase synchrony in every grey-matter voxel};
\node[capt,anchor=west,text=ink] at (1.15,15.5) {Illustrated with recorded signals; every usable run and every grey-matter voxel are processed in the same way};
% lanes
\fill[rounded corners=3pt,black!4] (.3,10.95) rectangle (14.55,15.2);
\fill[rounded corners=3pt,paleOrange] (.3,6.6) rectangle (14.55,10.8);
\node[rotate=90,font=\bfseries\fontsize{9}{10}\selectfont,text=ink] at (.62,13.2) {EGG};
\node[rotate=90,font=\bfseries\fontsize{9}{10}\selectfont,text=fed] at (.62,8.8) {BOLD};
% EGG lane
\node[anchor=north] (e1) at (3.05,14.7) {\includegraphics[width=3.9cm]{egg_raw.pdf}};
\node[anchor=north] (e2) at (7.65,14.7) {\includegraphics[width=3.9cm]{egg_psd.pdf}};
\node[anchor=north] (e3) at (12.25,14.7) {\includegraphics[width=3.9cm]{egg_filt.pdf}};
\node[lab,text=ink,anchor=south] at (e1.north) {Raw EGG};
\node[lab,text=ink,anchor=south] at (e2.north) {Find the gastric peak};
\node[lab,text=ink,anchor=south] at (e3.north) {Band-pass, resample to TR};
\node[capt,anchor=north] at (e1.south) {3 channels; runs kept only\\after manual review};
\node[capt,anchor=north] at (e2.south) {peak f$_0$ in 2--4 cpm (about 3 on average)};
\node[capt,anchor=north] at (e3.south) {zero-phase FIR filter;\\resampled to TR (2 s)};
% BOLD lane
\node[anchor=north] (b1) at (3.05,10.25) {\includegraphics[width=2.3cm]{bold_vox.pdf}};

\node[anchor=north] (b3) at (12.25,10.25) {\includegraphics[width=3.9cm]{bold_filt.pdf}};
\node[lab,text=fed,anchor=south] at (b1.north) {Preprocessed BOLD};

\node[lab,text=fed,anchor=south] at (b3.north) {Same band-pass};
\node[capt,anchor=north] at (b1.south) {AFNI preprocessing, censored volumes\\interpolated; MNI, 3 mm (one voxel shown)};
\node[capt,anchor=north] at (b3.south) {same filter, 30 s trimmed at each end;\\faint: before filtering};
% flow within lanes
\draw[flow] (e1.east) -- (e2.west); \draw[flow] (e2.east) -- (e3.west);
\coordinate (bm) at (0,8.95); \draw[flow] (b1.east |- bm) -- (b3.west |- bm);
% merge both lanes into the Hilbert step
\coordinate (bus) at (14.95,0);
\draw[line width=1.6pt,draw=navy!80] (e3.east) -- (e3.east -| bus);
\draw[line width=1.6pt,draw=navy!80] (b3.east |- bm) -- (bm -| bus);
\draw[line width=1.6pt,draw=navy!80] (e3.east -| bus) -- (bm -| bus);
\node[anchor=north] (h) at (17.3,14.7) {\includegraphics[width=3.85cm]{phase.pdf}};
\node[lab,text=band,anchor=south] at (h.north) {Hilbert transform};
\draw[flow] (e3.east -| bus) -- (e3.east -| h.west);
\node[eqbox=navy,text width=4.0cm,font=\fontsize{7.6}{9.4}\selectfont,anchor=north] (pl) at (17.3,10.3)
  {\textbf{Synchrony in each voxel}\\[3pt]
   $\mathrm{PLV}=\Bigl|\tfrac{1}{T}\sum_{t} e^{\,i\,\Delta\phi(t)}\Bigr|$\\[4pt]
   $\Delta\phi=\phi_B-\phi_E$\\
   0 = no synchrony\\1 = constant phase difference};
\draw[flow] (h.south) -- (pl.north);

% ------------------------------------------------ C chance and prominence
\draw[panel] (0,-1.45) rectangle (19.6,6.1);
\node[badge] at (.55,5.6) {C}; \node[ptitle] at (1.15,5.6) {Above chance? Counting voxels that exceed their chance threshold};
\node[anchor=north] (c1) at (3.35,4.45) {\includegraphics[width=5.0cm]{chance.pdf}};
\node[lab,text=ink,anchor=south] at (c1.north) {Chance in each voxel};
\node[capt,anchor=north] at (c1.south) {Counted if PLV with the own EGG\\is above the 95th percentile of PLVs\\with other participants' EGG};
\node[anchor=north] (c2) at (10.0,4.4) {\includegraphics[width=6.2cm]{prom.pdf}};
\node[lab,text=ink,anchor=south] at ($(c2.north)+(0,.05)$) {Repeat in every grey-matter voxel};
\node[capt,anchor=north] at (c2.south) {voxels above chance (orange), shown for one run;\\40,000--46,000 grey-matter voxels per run};
\node[eqbox=fed,text width=4.5cm,font=\fontsize{7.8}{9.6}\selectfont,anchor=north] (c3) at (16.75,4.8)
  {\textbf{Fraction of voxels above chance}\\[2pt]
   the fraction of grey-matter voxels\\whose synchrony exceeds their own\\chance threshold\\[3pt]
   $\dfrac{1}{V}\sum_{v=1}^{V}\mathbf{1}\bigl[\mathrm{PLV}_v>P_{95}(\mathrm{chance}_v)\bigr]$\\[4pt]

   chance level: each other\\participant's EGG run in turn\\treated as the own EGG};
\draw[flow] (c1.east) -- (c2.west); \draw[flow] (c2.east) -- (c3.west |- c2.east);
\node[capt,text=ink] at (9.8,-1.05) {This fraction was compared between visits with linear mixed models (covariates: run, session order, head motion).};
\end{tikzpicture}
\end{document}
'''
tex = os.path.join(WORK, 'fig_methods.tex')
open(tex, 'w').write(TEX)
subprocess.run(['tectonic', '--chatter', 'minimal', 'fig_methods.tex'], cwd=WORK, check=True)
pdf = os.path.join(WORK, 'fig_methods.pdf')
for suffix in ('', '_nocap'):
    shutil.copy(pdf, os.path.join(OUT, f'fig_methods{suffix}.pdf'))
    subprocess.run(['pdftoppm', '-r', '300', '-png', '-singlefile', pdf, os.path.join(OUT, f'fig_methods{suffix}')],
                   check=True)
store_caption(OUT, 'fig_methods', 'Figure 1',
              'Design and analysis. (A) Fasted and fed morning visits in counterbalanced order, each with a blood '
              'glucose test and two 15-min resting-state fMRI runs with EGG. (B) The EGG is band-pass filtered at its '
              'gastric peak (f0 ± 0.015 Hz), each grey-matter voxel\'s BOLD signal in the same band, and the '
              'phase-locking value (PLV) is computed from Hilbert phases. (C) A voxel is synchronized when its PLV '
              'with the same-run EGG exceeds the 95th percentile of its PLVs with other participants\' EGG.')
print(f'Figure 1 written (prominence of the example run {PROM:.1f}%)')
