"""
Chamfer-distance evaluation for GaussianPlant branch structure.

Primary metric of the project: how well the recovered branch structure matches the
ground-truth dense branch point cloud (``point_cloud_branch_dense.ply``).

Two prediction sources are supported (both compared against the same GT):
  1. StrPr cylinder mesh  -> densely sampled on the surface  (``--cylinder *.ply``)
  2. Bounded AppGS branch points (``branch.ply``)             (``--points *.ply``)

A *.ply that contains faces is treated as a mesh (uniformly sampled); a *.ply with
only vertices is treated as a point cloud.

Usage
-----
# single prediction
python eval_chamfer.py \
    --gt /mnt/data/gaussianplant_data/newplant1/feature_pretrain/point_cloud/iteration_30000/point_cloud_branch_dense.ply \
    --cylinder output/newplant1/branch_cylinder.ply \
    --points   output/newplant1/branch.ply

# scan every run folder under a model dir and report the best one
python eval_chamfer.py --gt <gt.ply> --scan output/newplant1
"""
import argparse
import glob
import os

import numpy as np
import open3d as o3d
import torch


def load_ply_points(path, n_sample=100_000):
    """Load a ply as an (N,3) float64 numpy array.

    If the ply is a triangle mesh (has faces) it is uniformly sampled, so that a
    cylinder *mesh* and a dense *point cloud* are compared on equal footing.
    """
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=n_sample)
        return np.asarray(pcd.points)
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        raise ValueError(f"{path} has neither faces nor points")
    return pts


@torch.no_grad()
def chamfer(a, b, device="cuda", chunk=4096):
    """Symmetric chamfer distance between point sets a (Na,3) and b (Nb,3).

    Returns a dict with both directions and the symmetric sum, for the squared
    metric (pytorch3d convention) and the plain L2 metric (world units).
    """
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    b = torch.as_tensor(b, dtype=torch.float32, device=device)

    def nn_sq_dists(src, dst):
        out = torch.empty(src.shape[0], device=device)
        for i in range(0, src.shape[0], chunk):
            d = torch.cdist(src[i : i + chunk], dst)  # (chunk, Nb)
            out[i : i + chunk] = d.min(dim=1).values ** 2
        return out

    d_ab = nn_sq_dists(a, b)  # a -> b
    d_ba = nn_sq_dists(b, a)  # b -> a
    return {
        "cd_sq": (d_ab.mean() + d_ba.mean()).item(),
        "cd_sq_a2b": d_ab.mean().item(),
        "cd_sq_b2a": d_ba.mean().item(),
        "cd_l2": (d_ab.sqrt().mean() + d_ba.sqrt().mean()).item(),
        "cd_l2_a2b": d_ab.sqrt().mean().item(),
        "cd_l2_b2a": d_ba.sqrt().mean().item(),
    }


def eval_one(gt_pts, pred_path, n_sample, device):
    pred_pts = load_ply_points(pred_path, n_sample=n_sample)
    res = chamfer(pred_pts, gt_pts, device=device)
    res["n_pred"] = pred_pts.shape[0]
    return res


def fmt(res):
    return (
        f"CD(L2)={res['cd_l2']:.6f}  (pred->gt {res['cd_l2_a2b']:.6f}, "
        f"gt->pred {res['cd_l2_b2a']:.6f})  CD(sq)={res['cd_sq']:.6e}  "
        f"n_pred={res['n_pred']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="ground-truth dense branch ply")
    ap.add_argument("--cylinder", default=None, help="StrPr cylinder mesh ply")
    ap.add_argument("--points", default=None, help="bounded AppGS branch points ply")
    ap.add_argument("--scan", default=None,
                    help="recursively find branch_cylinder.ply / branch.ply under this dir and rank")
    ap.add_argument("--n_sample", type=int, default=100_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gt_pts = load_ply_points(args.gt, n_sample=args.n_sample)
    print(f"GT: {args.gt}  n_gt={gt_pts.shape[0]}")

    if args.cylinder:
        print(f"[cylinder] {args.cylinder}\n  {fmt(eval_one(gt_pts, args.cylinder, args.n_sample, args.device))}")
    if args.points:
        print(f"[points]   {args.points}\n  {fmt(eval_one(gt_pts, args.points, args.n_sample, args.device))}")

    if args.scan:
        rows = []
        for name in ("branch_cylinder.ply", "branch.ply"):
            for p in sorted(glob.glob(os.path.join(args.scan, "**", name), recursive=True)):
                try:
                    res = eval_one(gt_pts, p, args.n_sample, args.device)
                    rows.append((res["cd_l2"], p, res))
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {p}: {e}")
        rows.sort(key=lambda r: r[0])
        print(f"\n=== ranked by CD(L2), best first  ({len(rows)} files) ===")
        for cd, p, res in rows:
            print(f"{cd:.6f}  {os.path.relpath(p, args.scan)}  | {fmt(res)}")
        if rows:
            print(f"\nBEST: {rows[0][1]}")


if __name__ == "__main__":
    main()
