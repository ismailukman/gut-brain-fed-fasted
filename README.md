# GUT-BFF: stomach-brain phase synchrony in fed and fasted states

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![Modality](https://img.shields.io/badge/data-EGG%20%2B%20resting--state%20fMRI-6baed6)
![Preprocessing](https://img.shields.io/badge/preprocessing-AFNI-08306b)
![Status](https://img.shields.io/badge/status-research%20code-lightgrey)

Analysis code for the Gut-Brain Fed/Fasted (GUT-BFF) study, which compares synchrony between the gastric rhythm,
recorded with electrogastrography (EGG), and resting-state BOLD fMRI after an overnight fast and after breakfast in
the same participants.

## Overview

- **Design:** 25 healthy adults, two morning visits in counterbalanced order (fasted, fed), each with two 15-min
  resting-state runs and concurrent three-channel EGG on a 3T Philips Ingenia Elition scanner.
- **Synchrony:** phase-locking value (PLV) and amplitude-weighted PLV (awPLV) between the EGG and BOLD signals,
  both band-passed around each run's EGG peak (about 0.05 Hz).
- **Chance level:** each run's BOLD signal paired with the EGG of every other participant.
- **Outcomes:** coupling strength (synchrony above chance in five predefined region sets) and prominence (the share
  of grey-matter voxels whose PLV exceeds the 95th percentile of their own chance distribution).
- **Statistics:** linear mixed models with run, session order and head motion, checked with participant
  fixed-effects models; FDR correction across regions and networks; permutation tests for whole-brain maps.

The data are not included in this repository.

## Repository layout

```
src/
  config.py                 paths (set with environment variables)
  gutsync.py                filtering, Hilbert transform, PLV, model fitting
  surface_render.py         cortical surface figures
  01-02                     preprocessing inventory and data-quality metrics
  03                        atlases and region sets
  04                        state manipulation and data quality
  05-09                     synchrony, primary models, whole-brain tests, controls, glucose, reliability
  10-14                     head motion, network-level synchrony, both metrics, follow-up analyses
  15-18                     figures
```

## Installation

```
git clone git@github.com:ismailukman/gut-brain-fed-fasted.git
cd gut-brain-fed-fasted
pip install -r requirements.txt
```

AFNI preprocessing (`afni_proc.py`) is run beforehand; FreeSurfer's `fsaverage5` is used only for the surface figures.

## Inputs

Place these in the project folder (`GUTBFF_BASE`, by default the repository folder):

| Path | Content |
|---|---|
| `data/gutbff_analysis_manifest.csv` | one row per run: participant, visit, state, run, EGG review result, analysis set, path to the preprocessed BOLD file (`bold_file`) and to the filtered EGG (`egg_filtered_file`) |
| `data/gutbff_runs.csv` | one row per run: session order, EGG peak frequency and quality, mean framewise displacement, censoring, glucose, path to the AFNI motion parameters (`motion_file`) |
| `data/gutbff_subjects.csv` | one row per participant: analysis set, session order, age, sex, BMI, glucose at each visit, days between visits |
| `atlas_src/` | source atlases for `03_build_atlas.py` (below) |

EGG and motion paths in the tables are relative to `GUTBFF_RAW`. Each filtered EGG (`gast_data_*.npy`, 10 Hz) is
stored with its filter frequency (`max_freq*.npy`) in the same folder.

| Atlas file (`atlas_src/`) | Source |
|---|---|
| `Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz`, `Schaefer2018_200Parcels_7Networks_order.txt` | Schaefer et al. (2018), CBIG repository |
| `Tian_Subcortex_S1_3T_2009cAsym.nii.gz` | Tian et al. (2020), Melbourne Subcortex Atlas |
| `atlas.nii` | Harvard-Oxford atlas as distributed with the CONN toolbox (insula labels 3 and 4) |
| `networks.nii`, `networks.txt` | CONN toolbox network masks (32 ROIs, 8 networks) |
| `rebollo2018_gastric_network_mask.nii.gz` | Rebollo et al. (2018), NeuroVault image 51888 |

## Configuration

```
export GUTBFF_BASE=/path/to/project                # data/, atlas_src/; results and figures are written here
export GUTBFF_RAW=/path/to/raw                     # start of the relative EGG and motion paths
export GUTBFF_AFNI_DIR=/path/to/afni_output        # afni_proc output, for 01_inventory_afni.py
export GUTBFF_GRID_REF=/path/to/one_errts.nii.gz   # defines the 3-mm analysis grid
export FREESURFER_HOME=/path/to/freesurfer
```

## Running

Run the scripts in `src/` in numerical order, for example `python src/05_synchrony.py`.

| Script | Purpose |
|---|---|
| `01_inventory_afni.py`, `02_nuisance_metrics.py` | Preprocessing outputs; censoring, global correlation and temporal SNR per run |
| `03_build_atlas.py` | Resample atlases to the BOLD grid; define the five region sets |
| `04_state_and_quality.py` | Glucose, EGG usability and data quality, fed vs fasted |
| `05_synchrony.py` | PLV and awPLV per region set, parcel and voxel, with the chance level (slow: reads every BOLD run) |
| `06_primary_and_localization.py` | Coupling strength and prominence models, sensitivity analyses, region sets, overlap with Rebollo et al. (2018) |
| `07_voxel_second_level.py` | Whole-brain voxel-wise sign-flip permutation test |
| `08_prominence_control.py` | Chance level of prominence from mismatched EGGs (slow) |
| `09_control_glucose_reliability.py` | Prominence control statistics, glucose models, reliability (ICC) |
| `10_motion_egg.py` | Synchrony between the EGG and head motion, by state |
| `11_network_synchrony.py` | Network-level synchrony for CONN networks and region sets (slow; `--stats-only` reuses saved values) |
| `12_voxel_maps_plv_awplv.py` | Voxel maps of PLV and awPLV minus the chance mean (slow) |
| `13_primary_plv_awplv.py` | Primary outcomes with both metrics |
| `14_followups.py` | Cluster-based whole-brain test and comparison of region summaries (exploratory) |
| `15`–`18_figures_*.py` | Figures; set `GUTBFF_SEND=1` for the version without the Rebollo regions |

Tables are written to `results/` (CSV) and figures to `figures/v2/` (or `figures/send/`). Each figure is saved with
and without its caption.

## References

- Rebollo I, et al. Stomach-brain synchrony reveals a novel, delayed-connectivity resting-state network in humans. *eLife* 2018;7:e33321.
- Choe AS, et al. Phase-locking of resting-state brain networks with the gastric basal electrical rhythm. *PLoS One* 2021;16:e0244756.
- Levakov G, Ganor S, Avidan G. Reliability and validity of brain-gastric phase synchronization. *Hum Brain Mapp* 2023;44:4956-4966.
