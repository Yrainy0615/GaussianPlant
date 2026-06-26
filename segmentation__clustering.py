#!/usr/bin/env python
import argparse
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def compute_pca_features(points):
    if points.shape[0] < 3:
        eigvals = np.array([1.0, 0.5, 0.1])
        eigvecs = np.eye(3)
    else:
        centered = points - points.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / (points.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

    l1, l2, l3 = np.clip(eigvals, 1e-12, None)
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    scattering = l3 / l1
    anisotropy = (l1 - l3) / l1

    main_axis = eigvecs[:, 0]
    proj = points @ main_axis
    length = proj.max() - proj.min()
    thickness = np.sqrt(l3)

    return {
        "linearity": float(linearity),
        "planarity": float(planarity),
        "scattering": float(scattering),
        "anisotropy": float(anisotropy),
        "length": float(length),
        "thickness": float(thickness),
    }


def classify_cluster_as_branch(
    feat,
    n_points,
    min_points=50,
    linearity_th=0.6,
    length_th=0.05,
    lt_ratio_th=8.0,
):
    """
    返回 True 表示判为 branch，False 表示 leaf/其他。
    """
    if n_points < min_points:
        return False

    linearity = feat["linearity"]
    length = feat["length"]
    thickness = max(feat["thickness"], 1e-6)
    lt_ratio = length / thickness

    if (linearity > linearity_th and
        length > length_th and
        lt_ratio > lt_ratio_th):
        return True

    return False


def extract_branch_points(
    points,
    eps=0.02,
    min_samples=20,
    **cls_kwargs
):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = db.fit_predict(points)

    n_clusters = cluster_ids.max() + 1
    branch_mask = np.zeros(points.shape[0], dtype=bool)

    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue
        pts_c = points[idx]
        feat = compute_pca_features(pts_c)
        is_branch = classify_cluster_as_branch(
            feat,
            n_points=idx.size,
            **cls_kwargs
        )
        if is_branch:
            branch_mask[idx] = True

    # 噪声点（cluster_id = -1）默认丢弃
    return branch_mask


def main():
    parser = argparse.ArgumentParser(
        description="DBSCAN-based branch-only extraction baseline"
    )
    parser.add_argument("--input_ply", type=str, required=True,
                        help="Input point cloud (.ply)")
    parser.add_argument("--output_ply", type=str, required=True,
                        help="Output branch-only point cloud (.ply)")
    parser.add_argument("--eps", type=float, default=0.02,
                        help="DBSCAN eps")
    parser.add_argument("--min_samples", type=int, default=20,
                        help="DBSCAN min_samples")
    parser.add_argument("--min_points_cluster", type=int, default=50)
    parser.add_argument("--linearity_th", type=float, default=0.6)
    parser.add_argument("--length_th", type=float, default=0.05)
    parser.add_argument("--lt_ratio_th", type=float, default=8.0)

    args = parser.parse_args()

    print(f"[INFO] Load: {args.input_ply}")
    pcd = o3d.io.read_point_cloud(args.input_ply)
    if len(pcd.points) == 0:
        raise RuntimeError("Empty point cloud.")
    points = np.asarray(pcd.points, dtype=np.float32)

    branch_mask = extract_branch_points(
        points,
        eps=args.eps,
        min_samples=args.min_samples,
        min_points=args.min_points_cluster,
        linearity_th=args.linearity_th,
        length_th=args.length_th,
        lt_ratio_th=args.lt_ratio_th,
    )

    branch_points = points[branch_mask]
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)[branch_mask]
    else:
        colors = np.tile(np.array([[0.5, 0.25, 0.1]], dtype=np.float32),
                         (branch_points.shape[0], 1))

    pcd_branch = o3d.geometry.PointCloud()
    pcd_branch.points = o3d.utility.Vector3dVector(branch_points)
    pcd_branch.colors = o3d.utility.Vector3dVector(colors)

    print(f"[INFO] Branch points: {branch_points.shape[0]}")
    o3d.io.write_point_cloud(args.output_ply, pcd_branch, write_ascii=False)
    print(f"[INFO] Saved: {args.output_ply}")


if __name__ == "__main__":
    main()
