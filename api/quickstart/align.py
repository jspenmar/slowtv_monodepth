
import numpy as np

def align_median(pred: np.ndarray, target: np.ndarray) -> float:
    """Return scale factor for median-depth alignment."""
    return np.median(target)/np.median(pred)


def align_lsqr(pred: np.ndarray, target: np.ndarray) -> list[float, float]:
    """Return scale & shift factor for least-squares alignment."""
    A = np.array([[(pred ** 2).sum(), pred.sum()], [pred.sum(), pred.shape[0]]])
    if np.linalg.det(A) <= 0: return 0, 0  # Avoid singular matrices.

    b = np.array([(pred * target).sum(), target.sum()])
    x = np.linalg.inv(A) @ b
    return x.tolist()


def main():

    def to_inv(depth: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return (depth > 0) / (depth + eps)

    depth = np.load('.../kbr/file.npy')  # (h, w)
    lidar = np.load('.../lidar/file.npy')  # (h, w)
    valid = (lidar > 0) & (lidar < 100)

    depth_mask, lidar_mask = depth[valid], lidar[valid]

    # Median alignment.
    scale, shift = align_median(depth_mask, lidar_mask)
    depth, depth_mask = depth*scale, depth_mask*scale

    # Least-squares alignment happens in disparity space, not depth.
    disp, disp_mask = to_inv(depth), to_inv(depth_mask)
    scale, shift = align_lsqr(disp_mask, to_inv(lidar_mask))
    disp, disp_mask = disp*scale + shift, disp_mask*scale + shift
    depth, depth_mask = to_inv(disp), to_inv(disp_mask)
