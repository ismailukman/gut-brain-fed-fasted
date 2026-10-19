"""
Paths used by all scripts. Set them with environment variables; the defaults suit the layout described in README.md.

  GUTBFF_BASE        project folder with data/, atlas/, results/ and figures/ (default: the repository folder)
  GUTBFF_RAW         folder that the relative EGG and motion paths in data/*.csv start from (default: GUTBFF_BASE)
  GUTBFF_AFNI_DIR    afni_proc output folder, one sub-*/ses-*/ folder per visit (used by 01_inventory_afni.py)
  GUTBFF_ATLAS_SRC   folder with the source atlases (used by 03_build_atlas.py)
  GUTBFF_GRID_REF    one preprocessed BOLD file that defines the analysis grid (used by 03_build_atlas.py)
  FREESURFER_HOME    FreeSurfer installation; fsaverage5 is used for the surface figures
  GUTBFF_TEMPLATE    MNI 2009c T1 template, background for the subcortical slices in Figure 5
                     (default: AFNI's MNI152_2009_template.nii.gz in ~/abin)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.environ.get('GUTBFF_BASE', os.path.join(HERE, '..')))
RAW = os.path.abspath(os.environ.get('GUTBFF_RAW', BASE))
AFNI_DIR = os.environ.get('GUTBFF_AFNI_DIR', '')
ATLAS_SRC = os.environ.get('GUTBFF_ATLAS_SRC', os.path.join(BASE, 'atlas_src'))
GRID_REF = os.environ.get('GUTBFF_GRID_REF', '')
FSAVERAGE = os.path.join(os.environ.get('FREESURFER_HOME', '/Applications/freesurfer'), 'subjects', 'fsaverage5', 'surf')
TEMPLATE = os.environ.get('GUTBFF_TEMPLATE', os.path.expanduser('~/abin/MNI152_2009_template.nii.gz'))
