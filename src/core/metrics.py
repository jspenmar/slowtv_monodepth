from functools import wraps

import numpy as np
import sklearn.neighbors as skln
from scipy import ndimage

import src.typing as ty
from src.tools import BackprojectDepth, extract_edges, ops

__all__ = ['metrics_eigen', 'metrics_benchmark', 'metrics_pointcloud', 'metrics_ibims']


# HELPERS
# -----------------------------------------------------------------------------
def to_float(fn):
    """Helper to convert all metrics into floats."""
    @wraps(fn)
    def wrapper(*a, **kw):
        return {k: float(v) for k, v in fn(*a, **kw).items()}
    return wrapper
# -----------------------------------------------------------------------------


# EIGEN
# -----------------------------------------------------------------------------
@to_float
def metrics_eigen(pred: ty.A, target: ty.A) -> ty.Metrics:
    """Compute Kitti Eigen depth prediction metrics.
    From Eigen (https://arxiv.org/abs/1406.2283)

    NOTE: The `sq_rel` error is incorrect! The correct error is `((err_sq ** 2) / target**2).mean()`
    We use the incorrect metric for backward compatibility with the common Eigen benchmark.
    This metric has been incorrectly reported since the benchmark was introduced.

    :param pred: (ndarray) (n,) Masked predicted depth.
    :param target: (ndarray) (n,) Masked ground truth depth.
    :return: (dict) Computed depth metrics.
    """
    err = np.abs(pred - target)
    err_rel = err/target

    err_sq = err ** 2
    err_sq_rel = err_sq/target

    err_log_sq = (np.log(pred) - np.log(target)) ** 2

    thresh = np.maximum((target/pred), (pred/target))

    return {
        'AbsRel': err_rel.mean(),
        'SqRel': err_sq_rel.mean(),
        'RMSE': np.sqrt(err_sq.mean()),
        'LogRMSE': np.sqrt(err_log_sq.mean()),
        '$\\delta_{.05}$': 100*(thresh < 1.05).mean(),
        '$\\delta_{.1}$': 100*(thresh < 1.1).mean(),
        '$\\delta_{.25}$': 100*(thresh < 1.25).mean(),
        '$\\delta_{.25^2}$': 100*(thresh < 1.25 ** 2).mean(),
        '$\\delta_{.25^3}$': 100*(thresh < 1.25 ** 3).mean(),
    }
# -----------------------------------------------------------------------------


# BENCHMARK
# -----------------------------------------------------------------------------
@to_float
def metrics_benchmark(pred: ty.A, target: ty.A) -> ty.Metrics:
    """Compute Kitti Benchmark depth prediction metrics.
    From Kitti (https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_depth.zip devkit/cpp/evaluate_depth.cpp L19-120)

    Base errors are reported as `m`.
    Inv errors are reported as `1/km`.
    Log errors are reported as `100*log(m)`.
    Relative errors are reported as `%`.
    This roughly aligns the significant figures for all metrics.

    :param pred: (ndarray) (n,) Masked predicted depth.
    :param target: (ndarray) (n,) Masked ground truth depth.
    :return: (dict) Computed depth metrics.
    """
    err = np.abs(pred - target)  # Units: m
    err_sq = err ** 2

    err_inv = 1000 * np.abs(1/pred - 1/target)  # Units: 1/km
    err_inv_sq = err_inv ** 2

    # NOTE: This is a DIRECTIONAL error! This is required for the SI Log loss
    # Objective is to not penalize the prediction if the errors are consistently in the same direction.
    # I.e. if the prediction could be aligned by applying a constant scale factor.
    err_log = 100 * (np.log(pred) - np.log(target))  # Units: log(m)*100
    err_log_sq = err_log ** 2

    err_rel = 100 * (err/target)  # Units: %
    err_rel_sq = 100 * (err_sq/target**2)

    return {
        'MAE': err.mean(),
        'RMSE': np.sqrt(err_sq.mean()),
        'InvMAE': err_inv.mean(),
        'InvRMSE': np.sqrt(err_inv_sq.mean()),
        'LogMAE': np.abs(err_log).mean(),
        'LogRMSE': np.sqrt(err_log_sq.mean()),
        'LogSI': np.sqrt(err_log_sq.mean() - err_log.mean() ** 2),
        'AbsRel': err_rel.mean(),
        'SqRel': err_rel_sq.mean(),
    }
# -----------------------------------------------------------------------------


# POINTCLOUD
# -----------------------------------------------------------------------------
def _metrics_pts(pred: ty.A, target: ty.A, th: float) -> tuple[float, float]:
    """Helper to compute F-Score and IoU with different correctness thresholds."""
    P = (pred < th).mean()  # Precision - How many predicted points are close enough to GT?
    R = (target < th).mean()  # Recall - How many GT points have a predicted point close enough?
    if (P < 1e-3) and (R < 1e-3): return 0, 0  # No points are correct.

    f = 2*P*R / (P + R + 1e-5)
    iou = P*R / (P + R - (P*R) + 1e-5)
    return 100*f, 100*iou


def _chamfer_dist(pred: ty.T, target: ty.T) -> tuple[ty.T, ty.T]:
    """Helper to compute the chamfer distance to/from pred and target."""
    pred, target = ops.to_np((pred[0], target[0]))

    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=1, algorithm='kd_tree', n_jobs=1, metric='euclidean')
    nn_engine.fit(target)
    pred_nn = nn_engine.kneighbors(pred[::2], return_distance=True)[0].astype(np.float32).squeeze()

    nn_engine.fit(pred)
    target_nn = nn_engine.kneighbors(target[::2], return_distance=True)[0].astype(np.float32).squeeze()
    return pred_nn, target_nn


@to_float
def metrics_pointcloud(pred: ty.A, target: ty.A, mask: ty.A, K: ty.A) -> ty.Metrics:
    """Compute pointcloud-based prediction metrics.
    From Ornek: (https://arxiv.org/abs/2203.08122)

    These metrics are computed on the GPU, since Chamfer distance has quadratic complexity.
    Following the original paper, we set the default threshold of a correct point to 10cm.
    An extra threshold is added at 20cm for informative purposes, but is not typically reported.

    :param pred: (ndarray) (h, w) Predicted depth.
    :param target: (ndarray) (h, w) Ground truth depth.
    :param mask: (ndarray) (h, w) Mask of valid pixels.
    :param K: (ndarray) (4, 4) Camera intrinsic parameters.
    :return: (dict) Computed depth metrics.
    """
    device = ops.get_device('cpu')
    pred, target, K = ops.to_torch((pred, target, K), device=device)
    K_inv = K.inverse()[None]

    backproj = BackprojectDepth(pred.shape).to(device)
    pred_pts = backproj(pred[None, None], K_inv)[:, :3, mask.flatten()]
    target_pts = backproj(target[None, None], K_inv)[:, :3, mask.flatten()]

    pred_nn, target_nn = _chamfer_dist(pred_pts.permute(0, 2, 1), target_pts.permute(0, 2, 1))

    out = {'Chamfer': pred_nn.mean() + target_nn.mean()}

    for th in [0.05, 0.1, 0.2]:
        out[f'F-Score ({th*100:.0f})'], out[f'IoU ({th*100:.0f})'] = _metrics_pts(pred_nn, target_nn, th=th)

    return out
# -----------------------------------------------------------------------------


# EDGES
# -----------------------------------------------------------------------------
@to_float
def metrics_ibims(pred: ty.A, target: ty.A, mask: ty.A) -> ty.Metrics:
    """Compute edge-based prediction metrics.
    From IBIMS: (https://arxiv.org/abs/1805.01328v1)

    The main metrics of interest are the edge accuracy and completeness. However, we also provide the directed error.
    Edge accuracy measures how close the predicted edges are wrt the ground truth edges.
    Meanwhile, edge completeness measures how close the ground-truth edges are from the predicted ones.

    :param pred: (ndarray) (h, w) Predicted depth.
    :param target: (ndarray) (h, w) Ground truth depth.
    :param mask: (ndarray) (h, w) Mask of valid & edges pixels.
    :param K: (ndarray) (4, 4) Camera intrinsic parameters.
    :return: (dict) Computed depth metrics.
    """
    th_dir = 10  # Plane at 10 meters
    pred_dir = np.where(pred <= th_dir, 1, 0)
    target_dir = np.where(target <= th_dir, 1, 0)
    err_dir = pred_dir - target_dir

    th_edges = 10
    D_target = ndimage.distance_transform_edt(1 - mask)  # Distance of each pixel to ground truth edges

    pred_edges = extract_edges(pred, preprocess='log', sigma=1)
    D_pred = ndimage.distance_transform_edt(1 - pred_edges)  # Distance of each pixel to predicted edges
    pred_edges = pred_edges & (D_target < th_edges)  # Predicted edges close enough to real ones.

    return {
        'DirAcc': 100 * (err_dir == 0).mean(),  # Accurate order
        'Dir (-)': 100 * (err_dir == 1).mean(),  # Pred depth was underestimated
        'Dir (+)': 100 * (err_dir == -1).mean(),  # Pred depth was overestimated
        'EdgeAcc': D_target[pred_edges].mean() if pred_edges.sum() else th_edges,  # Distance from pred to target
        'EdgeComp': D_pred[mask].mean() if pred_edges.sum() else th_edges,  # Distance from target to pred
    }
# -----------------------------------------------------------------------------
