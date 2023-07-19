from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

import src.typing as ty
from src import LOGGER
from src.tools import TableFormatter, to_inv
from src.utils import MultiLevelTimer
from .metrics import metrics_benchmark, metrics_eigen, metrics_ibims, metrics_pointcloud

__all__ = ['MonoDepthEvaluator']


class MonoDepthEvaluator:
    """Class to evaluate depth predictions.

    NOTE: Least-squares alignment (from Midas) is computed in disparity space.

    :param metrics: (tuple[str]) List of metric collections to use. {benchmark, eigen, pointcloud, ibims}
    :param align_mode: (str|float) Strategy used to align predictions and ground-truth.
        If given a `float`, assume depth is metric (up to a know scale factor). {median, lsqr, 1, 5.4...}
    :param interp_mode: (str) Interpolation mode for upsampling predictions. {nearest, bilinear, bicubic}
    :param min: (float) Min ground-truth depth used for evaluation.
    :param max: (None|float) Max ground-truth depth used for evaluation. `None` to disable.
    :param use_eigen_crop: (bool) If `True`, enable classic Eigen cropping. Use only for legacy reasons.
    :param use_nyud_crop: (bool) If `True`, enable NYU-D border cropping. Use only with NYU-D.
    """
    def __init__(self,
                 metrics: ty.S[str] = ('benchmark', 'pointcloud'),
                 align_mode: ty.U[str, float] = 1,
                 interp_mode: str = 'bilinear',
                 min: float = 1e-3,
                 max: ty.N[float] = None,
                 use_eigen_crop: bool = False,
                 use_nyud_crop: bool = False,):
        self.align_mode = align_mode
        self.metrics = metrics
        self.min = min
        self.max = max
        self.use_eigen_crop = use_eigen_crop
        self.use_nyud_crop = use_nyud_crop
        self.interp_mode = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
        }[interp_mode]

    def __call__(self,
                 pred: ty.A,
                 target: ty.A,
                 metrics: ty.S[str],
                 K: ty.N[ty.A] = None,
                 mask: ty.N[ty.A] = None) -> ty.Metrics:
        """Compute metrics for a single item.

        :param pred: (ndarray) (h', w') Scaleless disparity prediction (potentially downsampled).
        :param target: (ndarray) (h, w) Ground-truth depth map.
        :param metrics: (list[str]) List of metric groups to compute.
        :param K: (None|ndarray) (4, 4) Camera intrinsic parameters.
        :param mask: (None|ndarray) (h, w) Additional mask indicating valid pixels.
        :return: (ty.Metrics) Computed metrics for the given item.
        """
        timer = MultiLevelTimer()

        with timer('Total'):
            with timer('Preprocess'):
                target = target.astype(np.float32)
                pred = to_inv(self.upsample(pred, target))

            with timer('Mask'):
                if mask is None: mask = np.ones_like(target, dtype=bool)
                mask &= self.get_mask(target) & (pred > 0)
                if mask.sum() == 0: return {}  # Skip if no valid pixels.

                pred_mask, target_mask = pred[mask], target[mask]
                if pred_mask.sum() == 0: return {}  # Skip if no valid depths.

            with timer('Scale'):
                inv = self.align_mode == 'lsqr'  # Least-squares alignment is computed in disparity space.
                a, b = self.align(pred_mask, target_mask, inv=inv)
                pred, pred_mask = self.scale(pred, a, b, inv=inv), self.scale(pred_mask, a, b, inv=inv)

            with timer('Metrics'):
                ms = {'Scale': a, 'Shift': b}
                for m in metrics:
                    if m == 'eigen': ms |= metrics_eigen(pred_mask, target_mask)
                    elif m == 'benchmark': ms |= metrics_benchmark(pred_mask, target_mask)
                    elif m == 'pointcloud': ms |= metrics_pointcloud(pred, target, mask, K)
                    elif m == 'ibims': ms |= metrics_ibims(pred, target, mask)

        # print(timer)
        return ms

    def run(self,
            preds: ty.A,
            data: ty.ArrDict,
            nproc: ty.N[int] = None,
            chunks: int = 1,
            max_items: ty.N[int] = None) -> tuple[ty.Metrics, ty.S[ty.Metrics]]:
        """Compute metrics for the given predictions on the target dataset.

        :param preds: (ndarray) (b, h, w) Predicted disparities (scaleless).
        :param data:  (ArrDict) NPZ dict with ground-truth depths and optional intrinsics, edge maps and cats.
        :param nproc: (None|int) Number of processes to use. `None` to let OS determine it.
        :param chunks: (int) Chunk size of iterator given to each process
        :param max_items: (None|int) Maximum number of items to process. Used for testing/debugging a subset.
        :return: (
            mean_metrics: (ty.Metrics) Average metrics across the whole dataset.
            metrics: (list[ty.Metrics]) ty.Metrics for each item in the dataset.
        )
        """
        targets, Ks, edges = data['depth'], data.get('K'), data.get('edge')
        cats, subcats = data.get('cat'), data.get('subcat')
        LOGGER.info('Loaded targets...')
        del data

        if Ks is None and 'pointcloud' in self.metrics:
            raise ValueError(f"Missing intrinsics when computing pointcloud metrics!")

        if edges is None and 'ibims' in self.metrics:
            raise ValueError(f"Missing edge masks when computing IBIMS metrics!")

        if (a := len(preds)) != (b := len(targets)):
            raise ValueError(f"Non-matching preds and targets! ({a} vs. {b})")

        n = min(len(targets), max_items) if max_items else len(targets)
        preds, targets = preds[:n], targets[:n]

        metrics = self._run(preds, targets, [m for m in self.metrics if m != 'ibims'], Ks, nproc=nproc, chunks=chunks)
        if edges is not None:
            edge_metrics = self._run(preds, targets, self.metrics, Ks, edges, nproc=nproc, chunks=chunks)
            [m1.update({f'{k}-Edges': v for k, v in m2.items()}) for m1, m2 in zip(metrics, edge_metrics)]

        if cats is not None: self.add_cats(metrics, cats, subcats)

        if len(metrics) != n: raise ValueError(f'Non-matching metrics and targets! ({len(metrics)} vs. {n})')
        metrics = [m for m in metrics if m]  # Remove empty/skipped items.
        mean_metrics = self.average(metrics)

        self.summarize(mean_metrics)
        return mean_metrics, metrics

    def _run(self,
             preds: ty.A,
             targets: ty.A,
             metrics: ty.S[str],
             Ks: ty.N[ty.A] = None,
             masks: ty.N[ty.A] = None,
             nproc: ty.N[int] = None,
             chunks: ty.N[int] = 1) -> ty.S[ty.Metrics]:
        """Compute metrics for the given predictions and targets using multiprocessing."""
        n = len(preds)
        args = tqdm(zip(
            preds, targets,
            (metrics for _ in range(n)),
            [None]*n if Ks is None else Ks,
            [None]*n if masks is None else masks,
        ), desc='Metrics', total=n)
        with Pool(nproc) as p: out = list(p.starmap(self, args, chunksize=chunks))
        return out

    def summarize(self, mean_metrics: ty.Metrics) -> None:
        """Helper to report mean performance of the whole dataset."""
        LOGGER.info(f'Summarizing results...')
        print(TableFormatter.from_dict(mean_metrics).to_latex(precision=4))

    def upsample(self, pred: ty.A, target: ty.A) -> ty.A:
        """Helper to upsample the prediction to the full target resolution."""
        h, w = target.shape
        if pred.shape != target.shape: pred = cv2.resize(pred, (w, h), interpolation=self.interp_mode)
        return pred

    def get_mask(self, target: ty.A) -> ty.A:
        """Helper to mask ground-truth depth based on the selected range and Eigen crop."""
        mask = target > self.min
        if self.max: mask &= target < self.max
        if self.use_eigen_crop: mask &= self._get_nyud_mask(target.shape)
        if self.use_nyud_crop: mask &= self._get_eigen_mask(target.shape)
        return mask

    @staticmethod
    def _get_eigen_mask(shape: tuple[int, int]) -> ty.A:
        """Return valid pixels when using an Eigen mask."""
        h, w = shape
        crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w], dtype=int)
        mask = np.zeros((h, w), dtype=bool)
        mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        return mask

    @staticmethod
    def _get_nyud_mask(shape: tuple[int, int]) -> ty.A:
        """Return valid pixels when using an NYUD mask."""
        assert shape == (480, 640)
        mask = np.zeros(shape, dtype=bool)
        mask[45:471, 41:601] = 1
        return mask

    def align(self, pred: ty.A, target: ty.A, inv: bool = False) -> tuple[float, float]:
        """Return scale & shift parameters based on the alignment mode.

        :param pred: (ndarray) (n,) Predicted scaleless depth (masked to valid points).
        :param target: (ndarray) (n,) Target depth (masked to valid points).
        :param inv: (bool) If `True`, compute alignment parameters in disparity space.
        :return: (float, float) Scale and shift alignment parameters.
        """
        if inv: pred, target = to_inv(pred), to_inv(target)

        if self.align_mode == 'median': r, s = self._align_median(pred, target)
        elif self.align_mode == 'lsqr': r, s = self._align_lsqr(pred, target)
        else: r, s = self._align_metric(self.align_mode)

        return float(r), float(s)

    @staticmethod
    def _align_metric(factor: ty.N[float] = None) -> tuple[float, float]:
        """Return scale factor for metric alignment (up to a known `factor`)."""
        return factor or 1, 0

    @staticmethod
    def _align_median(pred: ty.A, target: ty.A) -> tuple[float, float]:
        """Return scale factor for median-depth alignment."""
        return np.median(target)/np.median(pred), 0

    @staticmethod
    def _align_lsqr(pred: ty.A, target: ty.A) -> tuple[float, float]:
        """Return scale & shift factor for least-squares alignment."""
        A = np.array([[(pred**2).sum(), pred.sum()], [pred.sum(), pred.shape[0]]])
        if np.linalg.det(A) <= 0: return 0, 0  # Avoid singular matrices.

        b = np.array([(pred*target).sum(), target.sum()])
        x = np.linalg.inv(A) @ b
        return x.tolist()

    def scale(self, pred: ty.A, scale: float, shift: float, inv: bool = False) -> ty.A:
        """Apply alignment parameters to the given prediction, as `ax + b`, in disparity space if needed."""
        if inv: pred = to_inv(pred)
        pred = scale*pred + shift
        if inv: pred = to_inv(pred)

        pred = pred.clip(self.min, self.max)
        return pred

    def add_cats(self, metrics: ty.S[ty.Metrics], cats: ty.S[str], subcats: ty.S[str]) -> None:
        """Append category information to each metric dict in-place."""
        LOGGER.info("Appending category information...")
        for m, cat, subcat in zip(metrics, cats, subcats):
            if m: m['Cat'], m['SubCat'] = str(cat), str(subcat)

    @staticmethod
    def average(metrics: ty.S[ty.Metrics]) -> ty.Metrics:
        """Compute the average metrics across each metric."""
        keys = (k for k, v in metrics[0].items() if isinstance(v, float))
        avg = {k: float(np.mean([d[k] for d in metrics if k in d])) for k in keys}
        return avg
