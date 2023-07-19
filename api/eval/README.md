# Evaluation

---

## Evaluating Models

The basic script for computing the evaluation metrics for a given model on a target dataset is:

```shell
DATASET=kitti_eigen_benchmark  # Replace with desired dataset
CKPT=last  # {best, last}
ALIGN_MODE=lsqr  # {lsqr, median, 1, 5.4}

python api/eval/eval_depth.py --align-mode ${ALIGN_MODE} --cfg-file cfg/eval/${DATASET}.yaml \
--ckpt /path/to/models/experiment/name/models/${CKPT}.ckpt \
--save-file /path/to/models/experiment/name/models/results/${DATASET}_${CKPT}_${ALIGN_MODE}.yaml
```

If you instead computed the prediction separately (e.g. using the instructions from the following section), instead 
replace `ckpt` with `pred-file`. 

```shell
DATASET=kitti_eigen_benchmark  # Replace with desired dataset
CKPT=last
ALIGN_MODE=lsqr 

python api/eval/eval_depth.py --align-mode ${ALIGN_MODE} --cfg-file cfg/eval/${DATASET}.yaml \
--ckpt /path/to/models/experiment/name/preds/${DATASET}_${CKPT}.npz \
--save-file /path/to/models/experiment/name/models/results/${DATASET}_${CKPT}_${ALIGN_MODE}.yaml
```


### External Baselines
If you instead want to compute metrics for one of the supervised baselines, replace the ckpt file with a string in the 
following format: `{BASELINE_NAME}.{VARIANT_NAME}`. 
Currently available baselines are:
* `midas.MiDaS`: Original MiDaS model using ResNeXt-101 backbone.
* `midas.DPT_Large`: DPT variant of MiDaS using a ViT-Large backbone.
* `midas.BEiT_L_512`: DPT variant using a BEiT-Large backbone.
* `newcrfs.indoor`: NewCRFs model trained on NYUD-v2.
* `newcrfs.outdoor`: NewCRFs model trained on Kitti.

### Ground-Truth Alignment
The `align-mode` parameter represents the strategy used when align the depth predictions to the ground-truth.
* `lsqr`: (Default) Least-squares alignment used by `MiDaS`.
* `median`: Traditional alignment using median depth scaling. 
* `1`: No alignment. Assumes predictions are metric depth.
* `factor`: Applies a fixed scaling factor to the predictions. Models trained with Kitti stereo data will commonly use 5.4.

### Other Parameters
* `cfg-model`: Used for loading legacy models, where the config saved with the model checkpoint is not compatible with the current format. 
* `overwrite`: If `1`, will overwrite existing saved metrics and will run the predictions even if the model hasn't finished training.
* `device`: PyTorch device on which to compute the predictions. 
* `nproc`: Number of processes used to compute the metrics. Leave empty to use all available processes.
* `max-items`: Max number of dataset items to evaluate. Useful for debugging. 
* `log`: Logging verbosity level.

---

## Computing Predictions

If you want to only compute the predictions for a given model, without evaluating them, you can instead use the following script:

```shell
DATASET=kitti_eigen_benchmark  # Replace with desired dataset
CKPT=last  # {best, last}

python api/eval/export_depth.py --cfg-file cfg/export/${DATASET}.yaml \
--ckpt /path/to/models/experiment/name/models/${CKPT}.ckpt \
--save-file /path/to/models/experiment/name/models/preds/${DATASET}_${CKPT}.npz
```

All other parameters are the same as in the previous section.

--- 

## Generating Results Tables

Once results have been computed for all models, you can generate the results tables using [this script](generate_tables.py).
Generally, it's easier to modify the code with the desired models, rather than passing them as command-line arguments.

Update the following sections in order to run with your desired models:

```python
root = MODEL_ROOTS[-1]  # Replace with desired model root

# Add or remove as desired.
splits = [
    'kitti_eigen_zhou',
    'syns_val',
    'mc',
]

# Replace with desired models.
for split in splits:
    fs, ms = list(zip(*[
        get_models(root, exp='kbr', dataset=split, mode='lsqr', ckpt='last', models='base'),
        get_models(root, exp='benchmark', split=split, mode='*', ckpt='best', models='garg monodepth2_MS diffnet_MS hrdepth_MS'),
        get_models(root, exp='midas', split=split, models='MiDaS DPT_Large DPT_BEiT_L_512'),
        get_models(root, exp='newcrfs', split=split, mode='lsqr'),
    ]))
```
