"""
Minimal cortical-surface rendering with matplotlib (no nilearn needed).

Volume values (MNI space) are sampled at five depths between the FreeSurfer fsaverage5 white and pial surfaces
(nearest voxel, mean over non-zero samples) and shown on the inflated surface, shaded by sulcal depth and a
simple light. fsaverage coordinates are close to MNI (a few mm), which is adequate for display.

surface_panel(ax, img, hemi, view, cmap, vmin, vmax, outline_img=None)
"""
import os
import warnings

import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates

from config import FSAVERAGE as FS
_cache = {}


def _mesh(hemi):
    if hemi not in _cache:
        inf, faces = nib.freesurfer.read_geometry(os.path.join(FS, f'{hemi}.inflated'))
        pial, _ = nib.freesurfer.read_geometry(os.path.join(FS, f'{hemi}.pial'))
        white, _ = nib.freesurfer.read_geometry(os.path.join(FS, f'{hemi}.white'))
        sulc = nib.freesurfer.read_morph_data(os.path.join(FS, f'{hemi}.sulc'))
        _cache[hemi] = (inf, faces, pial, white, sulc)
    return _cache[hemi]


def sample(img, hemi):
    """Vertex values: mean of non-zero samples at 5 depths from white to pial surface (NaN if none)."""
    _, _, pial, white, _ = _mesh(hemi)
    data = np.asanyarray(img.dataobj).astype(float)
    inv = np.linalg.inv(img.affine)
    vals = []
    for f in np.linspace(0, 1, 5):
        xyz = white + f * (pial - white)
        ijk = nib.affines.apply_affine(inv, xyz).T
        vals.append(map_coordinates(data, ijk, order=0, cval=0))
    v = np.array(vals)
    v[v == 0] = np.nan
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean(v, 0)


def surface_panel(ax, img, hemi, view, cmap='RdBu_r', vmin=-1, vmax=1, outline_img=None, alpha=1.0):
    inf, faces, _, _, sulc = _mesh(hemi)
    vv = sample(img, hemi)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fv = np.nanmean(vv[faces], 1)
    # base: sulcal shading (gyri light, sulci darker)
    base = np.where(sulc[faces].mean(1) > 0, 0.55, 0.78)
    tri = inf[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    azim = {('lh', 'lateral'): 180, ('lh', 'medial'): 0, ('rh', 'lateral'): 0, ('rh', 'medial'): 180}[(hemi, view)]
    light = np.array([np.cos(np.deg2rad(azim)), np.sin(np.deg2rad(azim)), 0.35])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
    rgb = np.repeat(base[:, None], 3, 1)
    norm = mcolors.Normalize(vmin, vmax)
    cm = matplotlib_cmap(cmap)
    has = np.isfinite(fv)
    rgb[has] = (1 - alpha) * rgb[has] + alpha * cm(norm(fv[has]))[:, :3]
    if outline_img is not None:
        ov = sample(outline_img, hemi) > 0
        ov = np.nan_to_num(ov).astype(bool)
        edge = ov[faces].any(1) & ~ov[faces].all(1)
        rgb[edge] = [0.05, 0.05, 0.05]
    rgb = np.clip(rgb * shade[:, None], 0, 1)
    pc = Poly3DCollection(tri, facecolors=rgb, edgecolors='none', linewidths=0)
    pc.set_clip_on(False)                          # the 3D axes clip to a square; inflated brains are wider
    ax.add_collection3d(pc)
    lo, hi = inf.min(0), inf.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1), zoom=1.75)
    ax.view_init(elev=0, azim=azim)
    ax.set_proj_type('ortho')
    ax.axis('off')
    return norm, cm


def matplotlib_cmap(name):
    import matplotlib
    return matplotlib.colormaps[name] if isinstance(name, str) else name
