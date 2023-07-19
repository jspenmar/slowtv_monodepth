import logging
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

import src.devkits.mapfreereloc as mfr
import src.typing as ty
from src import PRED_REG, find_model_file, trigger_preds
from src.core import MonoDepthEvaluator
from src.tools import ops, parsers, to_inv


def compute_preds(name: str,
                  cfg: dict,
                  ckpt: str,
                  cfg_model: ty.N[list[Path]],
                  device: ty.N[str],
                  overwrite: bool) -> None:
    """Compute predictions for a given dataset and network cfg.

    :param name: (str) Name used when saving predictions.
    :param cfg: (dict) Dataset cfg, following `MonoDepthModule` conventions.
    :param ckpt: (str) Model checkpoint to load. Either our checkpoint file or external model name. See docs.
    :param cfg_model: (None|list[Path]) Optional model cfgs when loading our legacy models.
    :param device: (str) Device on which to compute predictions.
    :param overwrite: (bool) If `True`, compute predictions even if model has not finished training.
    :return:
    """
    trigger_preds()
    model_type, model_name = ckpt.split('.', maxsplit=2)
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
        net = predictor.load_model(model_name)

    cfg.update({
        'shape': predictor.get_img_shape(cfg['type']),
        'as_torch': True, 'use_aug': False, 'log_time': False,
    })

    ds = parsers.get_ds({cfg.pop('type'): cfg})
    ds = list(ds.values())[0]
    dl = DataLoader(ds, batch_size=16, num_workers=8, collate_fn=ds.collate_fn, pin_memory=True)

    logging.info("Computing predictions...")
    pool = Pool()
    predictor.apply(net, dl, func=process_batch_preds, use_stereo_blend=False, device=device, name=name, pool=pool)
    pool.close()
    pool.join()


def process_batch_preds(batch: ty.BatchData, preds: ty.A, name: str, pool: Pool) -> None:
    """Align depth predictions and save files."""
    m = batch[2]

    files = [mfr.Item(*items).get_depth_file(name) for items in zip(m['mode'], m['scene'], m['seq'], m['stem'])]
    targets, preds = ops.to_np([batch[1]['depth'].squeeze(), preds.squeeze()], permute=False)

    args = zip(targets, preds, files)
    pool.map_async(process_single_pred, args)


def process_single_pred(args):
    """Upsample, align and save a single prediction."""
    target, pred, file = args
    pred = upsample(pred, target)
    pred = align(pred, target)
    save_depth_image(file, pred)


def upsample(pred: ty.A, target: ty.A) -> ty.A:
    """Upsample predictions to match target shape."""
    if pred.shape == target.shape: return pred

    h, w = target.shape
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred


def align(pred: ty.A, target: ty.A) -> ty.A:
    """Align predictions to ground-truth depth using least-squares and convert into depths."""
    mask = (target > 0) & (target < 100)
    scale, shift = MonoDepthEvaluator._align_lsqr(pred[mask], to_inv(target[mask]))
    pred = scale*pred + shift
    pred = to_inv(pred)
    return pred


def save_depth_image(path: str, depth: ty.A) -> None:
    """Save depth map in MapFreeReloc format (png with depth in mm)."""
    depth = (depth * 1000).astype(np.uint16)
    cv2.imwrite(str(path), depth)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=str, help='Model checkpoint to load.')
    parser.add_argument('--name', type=str, required=True, help='Name used when saving predictions.')
    parser.add_argument('--mode', type=str, default='val', help='Dataset split to evaluate on.')
    parser.add_argument('--depth-src', type=str, default='dptkitti', choices={'dptkitti', 'dptnyud'}, help='MapFreeReloc depth used as metric source.')
    parser.add_argument('--cfg-model', default=None, nargs='*', type=Path, help='Optional configs used to load model.')
    args = parser.parse_args()

    cfg = dict(type='mapfree', mode=args.mode, depth_src=args.depth_src, datum='image depth')
    compute_preds(args.name, cfg, args.ckpt, cfg_model=args.cfg_model, device='cuda', overwrite=False)
