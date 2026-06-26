"""
Drop-in replacements for the handful of PyTorch3D ops this project uses.

PyTorch3D is a heavy, build-from-source dependency; we only need quaternion
transforms, KNN and Chamfer distance, so we provide minimal equivalents here.
Conventions match PyTorch3D exactly:
  * quaternions are real-first (w, x, y, z) and assumed unit norm,
  * ``knn_points`` returns **squared** distances,
  * ``chamfer_distance`` returns (loss, normals_loss) with mean point/batch reduction.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  quaternion transforms                                                      #
# --------------------------------------------------------------------------- #
def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def standardize_quaternion(quaternions):
    """Force the real part non-negative (PyTorch3D convention)."""
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_invert(quaternion):
    """Conjugate of a unit quaternion (real-first)."""
    scaling = quaternion.new_tensor([1, -1, -1, -1])
    return quaternion * scaling


def quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_apply(quaternion, point):
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


# --------------------------------------------------------------------------- #
#  KNN  (matches pytorch3d.ops.knn_points: squared dists, namedtuple return)  #
# --------------------------------------------------------------------------- #
_KNN = namedtuple("KNN", "dists idx knn")


def knn_points(p1, p2, K=1, return_nn=False, chunk=65536, **kwargs):
    """K-nearest neighbours of each point in p1 within p2.

    Args:
        p1: (B, N, D) query points.
        p2: (B, M, D) reference points.
    Returns:
        namedtuple(dists=(B,N,K) squared dists, idx=(B,N,K) long, knn=(B,N,K,D) or None)
    """
    B, N, D = p1.shape
    M = p2.shape[1]
    K = min(K, M)
    dists = torch.empty(B, N, K, device=p1.device, dtype=p1.dtype)
    idx = torch.empty(B, N, K, device=p1.device, dtype=torch.long)
    for b in range(B):
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = torch.cdist(p1[b, s:e], p2[b])          # (chunk, M) euclidean
            kd, ki = torch.topk(d, K, dim=-1, largest=False)
            dists[b, s:e] = kd ** 2                      # squared, like pytorch3d
            idx[b, s:e] = ki
    knn = None
    if return_nn:
        knn = torch.gather(
            p2.unsqueeze(1).expand(B, N, M, D),
            2,
            idx.unsqueeze(-1).expand(B, N, K, D),
        )
    return _KNN(dists, idx, knn)


# --------------------------------------------------------------------------- #
#  Chamfer distance                                                           #
# --------------------------------------------------------------------------- #
def chamfer_distance(x, y, **kwargs):
    """Symmetric squared chamfer distance, mean point & batch reduction.

    Returns (loss, None) — the second element mirrors pytorch3d's normals loss.
    """
    d_xy = knn_points(x, y, K=1).dists[..., 0]   # (B, N) squared
    d_yx = knn_points(y, x, K=1).dists[..., 0]   # (B, M) squared
    loss = d_xy.mean() + d_yx.mean()
    return loss, None
