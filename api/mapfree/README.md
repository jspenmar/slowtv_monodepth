
# MapFreeReloc

Follow the instructions from the [MapFreeReloc GitHub](https://github.com/nianticlabs/map-free-reloc) to 
set up the code and anaconda environment. 

```shell
MAPFREE_ROOT=/path/to/map-free-reloc

cd /path/to
git clone https://github.com/nianticlabs/map-free-reloc && cd map-free-reloc

conda env create -f environment.yml
conda activate mapfree
```

Download the dataset to `/path/to/slowtv_monodepth/data` (required to compute the predictions)
and create a symlink to it in `map-free-reloc`:

```shell
mkdir ${MAPFREE_ROOT}/data
DATA_ROOT=/path/to/slowtv_monodepth/data

ln -s ${DATA_ROOT}/mapfree ${MAPFREE_ROOT}/data/mapfree  # Remember to use absolute paths!
```

---

## Computing Predictions

To compute the predictions, run the following scipt:

```shell
python api/mapfree/generate_preds.py --ckpt models/<EXP>/<NAME>/<SEED>/models/last.ckpt \
--name <EXP>_<NAME>_<SEED> --mode val --depth-src dptkitti 
```

The `ckpt` and `cfg-model` arguments are the same as those in the [evaluation procedure](../eval/README.md).
The `depth-src` is required to align the prediction with the reference baseline metric depths. 
You can choose from a DPT model finetuned on either Kitti or NYUD-v2. 

For the sake of compatibility, it is worth a `seed` parameter to the name even when computing predictions for the 
supervised baselines (MiDaS, DPT and NewCRFs). 

---

## Evaluating Predictions

The evaluate the predictions we run the original MapFreeReloc scripts. 
For convenience, we provide the `api/mapfree/evaluate.sh` script, which wraps changing to the correct directory and updating the python executable.
Feel free to comment out methods you don't need to evaluate and add your own.

This script should be run from the repository root, i.e. `/path/to/slowtv_monodepth`.
You will need to modify `MAPFREE_ROOT` and `MAPFREE_PY` to point to the correct locations.

---

## Saving Predictions

If you want to save the model's predictions (e.g. to allow others to use them in their submission), run the following:

```shell
find ./mapfree -name "*.<NAME1>.png" -o -name "*.<NAME2>.png" | tar -czvf mapfree_<NAME>_depth -T -
```

---

## Deleting Predictions

After running the evaluations you may want to delete the intermediate predictions. 
You can check the names of the models you have generated with:
```shell
ls ${MAPFREE_ROOT}/data/mapfree/val/*/seq0
```

You can then delete them with:
```shell
rm ${MAPFREE_ROOT}/data/mapfree/val/*/*/*.<NAME>.*
```

---