# Dataset Downloading & Preprocessing

Assume the following directory roots (change as required)
```shell
REPO_ROOT=/path/to/slowtv_monodepth
DATA_ROOT=${REPO_ROOT}/data
```

---

## Paths
Datasets are expected to be in `$REPO_ROOT/data` by default.
Path management is done in [`src/paths.py`](../../src/paths.py).
The format for the expected datasets is 
```python
datas: dict[str, str] = {
    'kitti_raw': 'kitti_raw_sync',
    'kitti_raw_lmdb': 'kitti_raw_sync_lmdb',
    'syns_patches': 'syns_patches',
}
```
where `keys` are the dataset identifiers in the `registry` and `values` are the stems of the path where the dataset is stored.
E.g. `kitti_raw` would be stored in `$REPO_ROOT/data/kitti_raw_sync`. 
This is where you can add new datasets if required.

If you wish to store datasets in other locations, create the file [`$REPO_ROOT/PATHS.yaml`](../../PATHS.yaml).
This file should not be tracked by Git, since it might contain sensitive information about your machine/server.
Populate the file with the following contents:

```yaml
# -----------------------------------------------------------------------------
MODEL_ROOTS: []

DATA_ROOTS:
  - /path/to/dataroot1
  - /path/to/dataroot2
  - /path/to/dataroot3
# -----------------------------------------------------------------------------
```
> **NOTE:** Multiple roots may be useful if training in an HPC cluster where data has to be copied locally.  

Replace the paths with the desired location(s). 
Once again, the dataset would be expected to be located in `/path/to/dataroot1/kitti_raw_sync`.
These roots should be listed in preference order, and the first existing dataset directory will be selected.

> **NOTE:** This procedure can be applied to change the default locations in which to find pretrained models. 
> Simply add the desired paths to `MODEL_ROOTS` instead.

---

## Splits
We provide the exact splits used in our experiments for each dataset. 
In most cases, this corresponds with the official splits provided by each dataset's authors or splits commonly used by the community. 
For datasets such as DDAD, TUM-RGBD and Mannequin Challenge we provide our own splits, as the original splits are too large.

> **NOTE:** Some of the [`splits`](splits) directories do not contain any files. 
> This is done to create all necessary dataset directories and make the following steps simpler. 

First copy the splits to the desired data root:
```shell
cd ${REPO_ROOT}
python api/splits/copy_splits.py ${DATA_ROOT}
```

--- 

## Download
> **BY DOWNLOADING EACH DATASET YOU AGREE TO ABIDE BY EACH CORRESPONDING LICENSE OR AGREEMENT.**

Each of the datasets used in our training and evaluations can be downloaded using the provided scripts.
This process will take a long time. 
All scripts should resume downloads if interrupted.

```shell
cd ${DATA_ROOT}/DDAD && ${REPO_ROOT}/api/data/download/ddad.sh; cd -

cd ${DATA_ROOT}/Diode && ${REPO_ROOT}/api/data/download/diode.sh; cd -

cd ${DATA_ROOT}/kitti_depth_benchmark && ${REPO_ROOT}/api/data/download/kitti_depth_benchmark.sh; cd -
cd ${DATA_ROOT}/kitti_raw_sync && ${REPO_ROOT}/api/data/download/kitti_raw_sync.sh; cd -

cd ${DATA_ROOT}/MannequinChallenge && ${REPO_ROOT}/api/data/download/mannequin.sh; cd -

cd ${DATA_ROOT}/mapfree && ${REPO_ROOT}/api/data/download/mapfree.sh; cd -

cd ${DATA_ROOT}/NYUD_v2 && ${REPO_ROOT}/api/data/download/nyud.sh; cd -

cd ${DATA_ROOT}/Sintel && ${REPO_ROOT}/api/data/download/sintel.sh; cd -

cd ${DATA_ROOT} && ${REPO_ROOT}/api/data/download/syns_patches.sh; cd -  

cd ${DATA_ROOT}/slow_tv && ${REPO_ROOT}/api/data/download/slow_tv.sh; cd -

cd ${DATA_ROOT}/TUM_RGBD && ${REPO_ROOT}/api/data/download/tum.sh; cd -
```

> **NOTE:** The ground-truth for SYNS-Patches isn't publicly available at the moment. 

> **NOTE:** The current script for Mannequin Challenge only downloads the info files with urls and timestamps.
> We currently don't have the scrips to download and extract the corresponding frames. 
> See https://google.github.io/mannequinchallenge/www/index.html for more info.

---

## Preprocessing

Some datasets require additional preprocessing: 


### **Kitti**
Copy improved depth maps from the updated benchmark and compute depth hints.
If you wish to train using proxy depth supervision from SGBM predictions, generate them using the following commands.
We follow the procedure from [DepthHints](https://arxiv.org/abs/1909.09051) and compute the hints using multiple hyperparameters and the min reconstruction loss.

> **NOTE:** Generating the hints takes quite a while, so only generate hints for the splits you are going actively use.

```shell
python api/data/preprocess/copy_kitti_depth_benchmark.py ${DATA_ROOT}/kitti_depth_benchmark ${DATA_ROOT}/kitti_raw_sync
#rm -rf ${DATA_ROOT}/kitti_depth_benchmark  # Optional

# This exports Kitti hints for ALL splits. Comment out the ones you don't need.
api/data/preprocess/compute_all_kitti_hints.sh ${DATA_ROOT}/kitti_raw_sync
```

### **Mannequin Challenge**
 Generate Colmap reconstructions to extract test depth maps. 

```shell
api/data/preprocess/compute_mannequin_depth.sh ${DATA_ROOT}/MannequinChallenge
```

### **NYUD**
Export image/depth from mat files.

```shell
python api/data/preprocess/export_nyud.py ${DATA_ROOT}/NYUD_v2
```

### **SlowTV** 
Export video frames, estimate Colmap intrinsics, decimate & generate splits.

```shell
python api/data/preprocess/export_slow_tv.py --n-proc 16
```

--- 

## Evaluation Targets
In this section we will generate the ground-truth depth targets used for evaluation.
The following script will export the targets for all datasets/splits. 
Comment out the ones you don't need.
```shell
api/data/export_gt/export_all.sh
```

Expected number of images: 
- DDAD (val): 1000
- Diode(val indoors): 325
- Diode(val outdoor): 446
- Kitti Eigen (test): 697
- Kitti Eigen Zhou (test): 700
- Kitti Eigen Benchmark (test): 652
- Mannequin Challenge (test): 1000
- Sintel (train): 1064
- SYNS Patches (val): 400
- SYNS Patches (test): 775
- NYUD_v2 (test): 654
- TUM_RGBD (test): 2500

---

## Generate LMDBs (Recommended)
Once you have finished downloading and preprocessing the datasets, you can optionally convert it into LMDB format.
This should make the data load faster, as well as reduce the load on the filesystem. 
You might find this beneficial if you are training in a cluster with limited bandwidth.

We only do this for training datasets (including their validation split): Kitti, Mannequin Challenge and SlowTV.

```shell
mkdir ${DATA_ROOT}/kitti_raw_sync_lmdb && python api/data/lmdb/kitti.py  --use-hints 1 --use-benchmark 1   # Change if needed to skip hints.

mkdir ${DATA_ROOT}/MannequinChallenge_lmdb && python api/data/lmdb/mannequin.py

mkdir ${DATA_ROOT}/MapFreeLReloc_lmdb && python api/data/lmdb/mannequin.py

mkdir ${DATA_ROOT}/slow_tv_lmdb && python api/data/lmdb/slow_tv.py
```

---
