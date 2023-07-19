from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from src import LOGGER, MODEL_ROOTS, find_model_file, set_logging_level
from src.core import HeavyLogger, MonoDepthModule
from src.utils import callbacks as cb, io


def main():
    # ARGUMENT PARSING
    # ------------------------------------------------------------------------------------------------------------------
    parser = ArgumentParser(description='Monocular depth trainer.')
    parser.add_argument('--cfg-files', '-c', type=Path, nargs='*', help='Path to YAML config files to load (default, override).')
    parser.add_argument('--ckpt-dir', '-o', default=MODEL_ROOTS[-1], type=Path, help='Root path to store checkpoint in.')
    parser.add_argument('--name', '-n', required=True, type=str, help='Model name for use during saving.')
    parser.add_argument('--version', '-v', default=0, type=int, help='Model version number for use during saving.')
    parser.add_argument('--seed', '-s', default=42, type=int, help='Random generator seed.')
    parser.add_argument('--gpus', '-g', default=1, type=int, help='Number of training GPUs.')
    parser.add_argument('--log', '-l', default='info', help='Logging verbosity level.')
    args = parser.parse_args()

    set_logging_level(args.log)

    LOGGER.info(f'Creating config from {[f"{f.parent.stem}/{f.name}" for f in args.cfg_files]}...')
    LOGGER.warning('Please ensure configs are in the correct order! (default, overwrite1, overwrite2...)')
    cfg = io.load_merge_yaml(*args.cfg_files)

    cfg['loader']['seed'] = args.seed
    cfg['loader']['use_ddp'] = args.gpus > 1
    # ------------------------------------------------------------------------------------------------------------------

    # LOGGER
    # ------------------------------------------------------------------------------------------------------------------
    version = f'{args.version:03}'
    save_dir = args.ckpt_dir/args.name/version
    io.mkdirs(save_dir)

    logger_type = cfg['trainer'].get('logger', 'wandb')
    if logger_type == 'tensorboard': logger = pll.TensorBoardLogger(
        save_dir=args.ckpt_dir, name=args.name, version=version, default_hp_metric=False
    )
    elif logger_type == 'wandb': logger = pll.WandbLogger(
        save_dir=save_dir, version=f'{args.name}_{version}', project=args.ckpt_dir.stem, log_model=False, resume=None
    )
    else:
        raise ValueError(f'Logger "{logger_type}" not supported. Please choose from "{{tensorboard, wandb}}"')
    # ------------------------------------------------------------------------------------------------------------------

    # CALLBACKS
    # ------------------------------------------------------------------------------------------------------------------
    monitor = cfg['trainer'].get('monitor', 'AbsRel')
    monitor = f'val_losses/{monitor}' if monitor == 'loss' else f'val_metrics/{monitor}'
    mode = 'max' if 'Acc' in monitor else 'min'
    cb_ckpt = plc.ModelCheckpoint(
        dirpath=save_dir/'models', filename='best',
        auto_insert_metric_name=False,  # Removes '=' from filename
        monitor=monitor, mode=mode,
        save_last=True, save_top_k=1, verbose=True,
    )

    cbks = [
        cb_ckpt,
        plc.LearningRateMonitor(logging_interval='epoch'),
        plc.RichModelSummary(max_depth=2),
        cb.RichProgressBar(),
        cb.TrainingManager(Path(cb_ckpt.dirpath)),
        cb.DetectAnomaly(),
        HeavyLogger(n_imgs=6, n_cols=2),
    ]

    if cfg['trainer'].get('swa'):  # FIXME: Not Tested!
        cbks.append(plc.StochasticWeightAveraging(swa_epoch_start=0.75, annealing_epochs=5, swa_lrs=cfg['optimizer']['lr']))

    if cfg['trainer'].get('early_stopping'):
        cbks.append(plc.EarlyStopping(monitor=monitor, mode=mode, patience=5))
    # ------------------------------------------------------------------------------------------------------------------

    # CREATE MODULE
    # ------------------------------------------------------------------------------------------------------------------
    pl.seed_everything(args.seed)

    # Load model weights from pretrained model.
    if path := cfg['trainer'].get('load_ckpt'):
        path = find_model_file(path)
        LOGGER.info(f'Loading model from checkpoint: {path}')
        # FIXME: You might need to change `strict=False` if the cfg of the base model is not compatible!
        model = MonoDepthModule.load_from_checkpoint(path, cfg=cfg, strict=True)
    else:
        model = MonoDepthModule(cfg)

    # Resume training from earlier checkpoint.
    resume_path = None
    if cfg['trainer'].get('resume_training'):
        LOGGER.info('Resuming training...')
        if (path := Path(cb_ckpt.dirpath, 'last.ckpt')).is_file(): resume_path = path
        else: LOGGER.warning(f'No previous checkpoint found in "{path.parent}". Beginning training from scratch...')
    # ------------------------------------------------------------------------------------------------------------------

    # CREATE TRAINER
    # ------------------------------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        devices=args.gpus, accelerator='gpu', strategy='auto',

        max_epochs=cfg['trainer']['max_epochs'],
        limit_train_batches=1., limit_val_batches=200,  # Limit val batches to speed up training.
        accumulate_grad_batches=cfg['trainer'].get('accumulate_grad_batches', 1),
        log_every_n_steps=cfg['trainer'].get('log_every_n_steps', 100),
        use_distributed_sampler=False,

        benchmark=cfg['trainer'].get('benchmark', False),
        precision=cfg['trainer'].get('precision', 32),
        gradient_clip_val=cfg['trainer'].get('gradient_clip_val', None),

        logger=logger, callbacks=cbks, enable_model_summary=False,
    )
    LOGGER.info(f'-> Number of training batches: {trainer.num_training_batches}...')
    # ------------------------------------------------------------------------------------------------------------------

    # FIT
    # ------------------------------------------------------------------------------------------------------------------
    if model.auto_scale_lr:
        LOGGER.info(f'Scaling LR "{model.lr}" using "{trainer.num_devices}" devices '
                    f'and accumulate "{trainer.accumulate_grad_batches}" batches...')
        model.lr *= trainer.num_devices * trainer.accumulate_grad_batches
    trainer.fit(model, ckpt_path=resume_path)
    # ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
