import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

import src.typing as ty
from src import find_model_file
from src.registry import PRED_REG, trigger_preds
from src.tools import parsers
from src.utils import io, set_logging_level


def save_preds(file: Path, preds: ty.A) -> None:
    """Helper to save network predictions to a NPZ file. Required for submitted to the challenge."""
    io.mkdirs(file.parent)
    logging.info(f"Saving network predictions to '{file}'...")
    np.savez_compressed(file, pred=preds)


def compute_preds(cfg: dict,
                  ckpt: str,
                  cfg_model: ty.N[list[Path]],
                  device: ty.N[str],
                  overwrite: bool) -> ty.A:
    """Compute predictions for a given dataset and network cfg.

    `ckpt` can be provided as:
        - Path: Path to a pretrained checkpoint trained using the benchmark repository.
        - Name: Name indicating the external model type and variant to load, e.g. midas.MiDaS, newcrfs.indoor.

    Currently supported external models are: {
        midas.{MiDaS, DPT_Large, DPT_BEiT_L_512},
        newcrfs.{indoor,outdoor},
    }

    :param cfg: (dict) Dataset cfg, following `MonoDepthModule` conventions.
    :param ckpt: (str) Model checkpoint to load. Either our checkpoint file or external model name. See docs.
    :param cfg_model: (None|list[Path]) Optional model cfgs when loading our legacy models.
    :param device: (str) Device on which to compute predictions.
    :param overwrite: (bool) If `True`, compute predictions even if model has not finished training.
    :return:
    """
    trigger_preds()
    model_type, name = ckpt.split('.', maxsplit=2)
    model_type = model_type if model_type in PRED_REG else 'ours'

    predictor = PRED_REG[model_type]()
    if model_type == 'ours':
        ckpt = find_model_file(ckpt)
        if not (ckpt.parent/'finished').is_file() and not overwrite:
            logging.error(f"Training for '{ckpt}' has not finished...")
            logging.error("Set `--overwrite 1` to run this evaluation anyway...")
            exit()

        logging.info(f"Loading pretrained model from '{ckpt}'")
        net = predictor.load_model(ckpt, cfg_model)
    else:
        net = predictor.load_model(name)

    cfg.update({
        'shape': predictor.get_img_shape(cfg['type']),
        'as_torch': True, 'use_aug': False, 'log_time': False,
    })

    ds = parsers.get_ds({cfg.pop('type'): cfg})
    ds = list(ds.values())[0]
    dl = DataLoader(ds, batch_size=12, num_workers=8, collate_fn=ds.collate_fn, pin_memory=True)

    logging.info("Computing predictions...")
    preds = predictor(net, dl, use_stereo_blend=False, device=device)
    return preds


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to export network predictions on a target dataset.')
    parser.add_argument('--cfg-file', required=True, type=Path, help='Path to dataset config to compute predictions for.')
    parser.add_argument('--ckpt', required=True, type=str, help='Model checkpoint name. Use `midas.DPT_Large` for external baselines.')
    parser.add_argument('--cfg-model', default=None, nargs='*', type=Path, help='ty.N configs used to load model. (Used primarily for legacy models)')
    parser.add_argument('--save-file', default=None, type=Path, help='Path to NPZ file to save predictions')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing prediction files.')
    parser.add_argument('--device', default=None, help='Device on which to compute predictions.')
    parser.add_argument('--log', default='info', help='Logging verbosity level.')
    args = parser.parse_args()

    set_logging_level(args.log)

    if args.save_file and args.save_file.is_file() and not args.overwrite:
        logging.error(f"Evaluation file already exists '{args.save_file}'...\n"
                      f"Set `--overwrite 1` to run this evaluation anyway...")
        exit()

    cfg = io.load_yaml(args.cfg_file)['dataset']
    preds = compute_preds(cfg, args.ckpt, args.cfg_model, args.device, args.overwrite)
    if args.save_file: save_preds(args.save_file, preds)
