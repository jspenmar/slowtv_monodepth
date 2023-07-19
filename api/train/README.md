# Training

---

## Train
To train a network, simply run

```shell
python api/train/train.py --cfg-files cfg/<EXP>/default.yaml cfg/<EXP>/<MODEL>.yaml  \
--ckpt-dir ./models/<EXP> --name <MODEL> --version 42 --seed 42
```

- `cfg-files`: List of training config files, containing details for the networks, losses, optimizers... Configs should be listed in reverse priority, i.e. `[default, override1, override2]`.
- `ckpt-dir`: Path to root directory to store checkpoints. Final checkpoints will be saved to `ckpt-dir/<NAME>/<VERSION>/models`
- `name`: Name used when saving models in `ckpt-dir`, as above. 
- `version`: Version used when saving in `ckpt-dir`, as above. Typically set to the same value as the random seed.
- `seed`: Random seed used to initialize the world. Note that reproducibility still isn't fully guaranteed.

For details on creating config files see [this README](../../src/README.md).
A good place to start is the [default config](../../cfg/default.yaml), which illustrates the various available parameters.

It is generally good practice to train multiple models (at least 3) with different random seeds.
Performance is then compared via the average metrics obtained by each instance of the contribution models.
Note that this behaviour is already incorporated into the [generate_tables](../eval/generate_tables.py) script.
For no reason in particular, I tend to train models using seeds `42`, `195` & `335`. 

### Checkpoints
By default, the training script saves a `best.ckpt` and `last.ckpt` checkpoints.

- `last.ckpt` is saved after every epoch and should be used when resuming interrupted training. 
- `best.ckpt` is the checkpoint for the model that produced the best performance throughout training. 

When training exclusively on Kitti, the 'best' performance is determined by tracking only one specific metric, typically `AbsRel`.
When some datasets don't contain ground-truth, we instead track validation loss. 
In practice, I've found that the validation loss does not necessarily reflect the best performance. 
As such, it is generally best to just evaluate the `last` model.

### Logging
By default, the training script logs progress to Weights & Biases. 
You'll have to log into your account to view the logs.
Scalars (e.g. losses and metrics) are logged every 100 steps, while images and other artifacts are only logged at the end of each epoch.

Training progress can instead be logged to TensorBoard by changing the config to:
    
```yaml
trainer:
    logger: 'tensorboard'
```

To monitor the performance run
```shell
tensorboard --bind_all --logdir ./models/<EXP>  # `--bind_all` for use when ssh tunnelling.
```

---

## Dev
We provide the similar [train_dev](./train_dev.py) for use when developing and debugging. 
The main differences w.r.t. [train](./train.py) are:

- Use of `TQDMProgressBar`, to allow for better debugging with `breakpoint()`.
- Checkpoints default to `/tmp` instead of `./models`.
- Defaults to 10 training epochs.
- Defaults to 10 train/val batches per epoch.
- Easy enabling of grad norm tracking and gradient anomaly detection.

---
