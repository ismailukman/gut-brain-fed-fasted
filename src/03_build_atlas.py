"""
Resample the atlases to the BOLD grid (MNI 2009c, 3 mm) and define the five network families.

  1 insular       Harvard-Oxford insular cortex (labels 3, 4 of the CONN atlas.nii)
  2 sensorimotor  Schaefer-200 SomMot parcels, insula excluded
  3 salience      Schaefer-200 SalVentAttn parcels, insula excluded
  4 default-mode  Schaefer-200 Default parcels, insula excluded
  5 subcortical   Tian S1 (16 regions)
Also: Schaefer-200 + Tian S1 parcels (216), the Rebollo et al. (2018) gastric network
(NeuroVault image 51888) and the 32 CONN network masks.
Outputs: atlas/*_3mm.nii.gz and label files

Usage: python 03_build_atlas.py
"""
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from config import ATLAS_SRC, BASE, GRID_REF
OUT = os.path.join(BASE, 'atlas')

FAMILIES = ['insular', 'sensorimotor', 'salience', 'default-mode', 'subcortical']


def resample_nn(src_img, ref_shape, ref_affine):
    """Nearest-neighbour resample of a label image onto a reference grid."""
    ijk = np.indices(ref_shape).reshape(3, -1)
    xyz = nib.affines.apply_affine(ref_affine, ijk.T)
    src_ijk = nib.affines.apply_affine(np.linalg.inv(src_img.affine), xyz).T
    data = np.asanyarray(src_img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    out = map_coordinates(data, src_ijk, order=0, mode='constant', cval=0)
    return out.reshape(ref_shape)


ref = nib.load(GRID_REF)
shape, aff = ref.shape[:3], ref.affine

schaefer = resample_nn(nib.load(os.path.join(ATLAS_SRC, 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz')), shape, aff).astype(int)
tian = resample_nn(nib.load(os.path.join(ATLAS_SRC, 'Tian_Subcortex_S1_3T_2009cAsym.nii.gz')), shape, aff).astype(int)
ho = resample_nn(nib.load(os.path.join(ATLAS_SRC, 'atlas.nii')), shape, aff).astype(int)
rebollo = resample_nn(nib.load(os.path.join(ATLAS_SRC, 'rebollo2018_gastric_network_mask.nii.gz')),
                      shape, aff) > 0

labels = {}
for line in open(os.path.join(ATLAS_SRC, 'Schaefer2018_200Parcels_7Networks_order.txt')):
    parts = line.split()
    labels[int(parts[0])] = parts[1]
net_of = {k: v.split('_')[2] for k, v in labels.items()}   # e.g. SomMot

parcels = schaefer.copy()
parcels[(tian > 0)] = 200 + tian[tian > 0]                 # Tian overrides any cortical spill-over
insula = np.isin(ho, [3, 4])

family = np.zeros(shape, dtype=np.int16)
for code, net in ((2, 'SomMot'), (3, 'SalVentAttn'), (4, 'Default')):
    ids = [k for k, v in net_of.items() if v == net]
    family[np.isin(schaefer, ids) & ~insula & (tian == 0)] = code
family[insula & (tian == 0)] = 1
family[tian > 0] = 5

conn_img = nib.load(os.path.join(ATLAS_SRC, 'networks.nii'))
conn = np.stack([resample_nn(nib.Nifti1Image(np.asanyarray(conn_img.dataobj)[..., i], conn_img.affine), shape, aff)
                 for i in range(conn_img.shape[-1])], -1).astype(np.int16)
conn_names = [l.split(' (')[0].strip() if l.count('(') == 1 else l.rsplit(' (', 1)[0].strip()
              for l in open(os.path.join(ATLAS_SRC, 'networks.txt')) if l.strip()]

os.makedirs(OUT, exist_ok=True)
nib.save(nib.Nifti1Image(conn, aff), os.path.join(OUT, 'conn_networks_3mm.nii.gz'))
with open(os.path.join(OUT, 'conn_networks_labels.txt'), 'w') as f:
    f.write('\n'.join(conn_names) + '\n')
for name, arr in (('families_3mm', family), ('parcels216_3mm', parcels.astype(np.int16)),
                  ('rebollo2018_3mm', rebollo.astype(np.int16))):
    nib.save(nib.Nifti1Image(arr, aff), os.path.join(OUT, f'{name}.nii.gz'))
with open(os.path.join(OUT, 'families_labels.txt'), 'w') as f:
    for i, n in enumerate(FAMILIES, 1):
        f.write(f'{i}\t{n}\t{(family == i).sum()} voxels\n')

for i, n in enumerate(FAMILIES, 1):
    print(f'{i} {n:13s} {(family == i).sum():5d} voxels | overlap with Rebollo: '
          f'{(rebollo & (family == i)).sum()} voxels')
print(f'parcels: {len(np.unique(parcels)) - 1} | Rebollo voxels on grid: {rebollo.sum()} '
      f'({(rebollo & (parcels > 0)).sum()} inside parcels)')
