from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import src.typing as ty
from export_preds import compute_preds
from src import LOGGER, set_logging_level
from src.core import MonoDepthEvaluator
from src.tools import parsers
from src.typing import Metrics
from src.utils.io import load_yaml, write_yaml


def save_metrics(file: Path, metrics: ty.U[Metrics, ty.S[Metrics]]):
    """Helper to save metrics."""
    LOGGER.info(f'Saving results to "{file}"...')
    file.parent.mkdir(exist_ok=True, parents=True)
    write_yaml(file, metrics, mkdir=True)


def compute_eval_metrics(preds: ty.A,
                         cfg_file: Path,
                         align_mode: ty.U[str, float],
                         nproc: ty.N[int] = None,
                         max_items: ty.N[int] = None) -> tuple[Metrics, ty.S[Metrics]]:
    """Compute evaluation metrics from scaleless network disparities (see `compute_eval_preds`).

    :param preds: (NDArray) (b, h, w) Precomputed unscaled network predictions.
    :param cfg_file: (Path) Path to YAML config file.
    :param align_mode: (str|float) Strategy used to align the predictions to the ground-truth. {median, lsqr, 1, 5.4...}
    :param nproc: (None|int) Number of processes to use. `None` to let OS determine it.
    :param max_items: (None|int) Maximum number of items to process. Used for testing/debugging a subset.
    :return: (
        mean_metrics: (Metrics) Average metrics across the whole dataset.
        metrics: (list[Metrics]) Metrics for each item in the dataset.
    )
    """
    cfg = load_yaml(cfg_file)
    cfg_ds, cfg_args = cfg['dataset'], cfg['args']
    try: cfg_args['align_mode'] = float(align_mode)  # Align mode is a fixed scalar factor.
    except (ValueError, TypeError): cfg_args['align_mode'] = align_mode

    target_stem = cfg_ds.pop('target_stem', f'targets_{cfg.get("mode", "test")}')
    ds = parsers.get_ds({cfg_ds.pop('type'): cfg_ds})
    ds = next(iter(ds.values()))
    target_file = ds.split_file.parent/f'{target_stem}.npz'

    LOGGER.info(f'Loading targets from "{target_file}"...')
    data = np.load(target_file, allow_pickle=True)
    mean_metrics, metrics = MonoDepthEvaluator(**cfg_args).run(preds, data, nproc=nproc, max_items=max_items)
    return mean_metrics, metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--align-mode', default='lsqr', type=str, help='Strategy to align preds and ground-truth (str or scaling factor). {median, lsqr, 1, 5.4...}')
    parser.add_argument('--cfg-file', required=True, type=Path, help='Path to YAML eval config.')
    parser.add_argument('--pred-file', default=None, type=Path, help='ty.N `npz` path to precomputed predictions.')
    parser.add_argument('--ckpt', default=None, type=str, help='ty.N path to model ckpt or external model to compute predictions.')
    parser.add_argument('--cfg-model', default=None, nargs='*', type=Path, help='ty.N configs used to load model.')
    parser.add_argument('--save-file', default=None, type=Path, help='Path to YAML file to save evaluation metrics.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing metrics files.')
    parser.add_argument('--device', default=None, help='Device on which to compute predictions.')
    parser.add_argument('--nproc', default=None, type=int, help='Number of processed used. If empty, determined by the OS.')
    parser.add_argument('--max-items', default=None, type=int, help='Max number of items to use in evaluation. If empty, use whole dataset')
    parser.add_argument('--log', default='info', help='Logging verbosity level.')
    args = parser.parse_args()

    set_logging_level(args.log)

    if args.save_file and args.save_file.is_file() and not args.overwrite:
        LOGGER.error(f"Evaluation file already exists '{args.save_file}'...\n"
                     f"Set `--overwrite 1` to run this evaluation anyway...")
        exit()

    if args.pred_file:
        LOGGER.info(f"Loading predictions from '{args.pred_file}'...")
        preds = np.load(args.pred_file)['pred']
    else:
        if not args.ckpt: raise ValueError("Must provide either a `--pred-file` with precomputed predictions "
                                           "or a `--ckpt to compute predictions from!")
        cfg = load_yaml(args.cfg_file)['dataset']
        cfg.pop('target_stem', None)

        preds = compute_preds(cfg, args.ckpt, args.cfg_model, args.device, args.overwrite)

    mean_metrics, metrics = compute_eval_metrics(preds, args.cfg_file, args.align_mode, args.nproc, args.max_items)
    if args.save_file: save_metrics(args.save_file, mean_metrics)
