#!/usr/bin/env python
"""
Standalone semantic branch/leaf classifier for a *cleaned* plant point cloud.

Purely semantic: it classifies each Gaussian by the cosine similarity of its 128-d
DINOv3 feature (stored as `semantic_0..127` in the feature-3DGS ply) to the
`branch` / `leaf` DINOv3 *text* features in `dinov3_text_feats.pth`. No colour, no
geometry — meant for cases where the colour cue fails (e.g. yellow / autumn leaves).

The text prompts were computed offline and stored as `text_feats_dim128` of shape
[4, 128]. The index → prompt mapping is not fully documented in the repo (the scene
loader comments imply `0=stem, 1=leaf, 2=background, 3=plant`), so `--branch_idx` /
`--leaf_idx` are exposed and the script prints the branch fraction for every prompt
pair so you can pick the right one on your own data.

Outputs (under --out):
  <name>_label.ply     points + a `label` scalar (branch probability in [0,1])
  <name>_color.ply     points RGB-coloured  red = branch, green = leaf
  <name>_semcls.png    3-view scatter of the coloured classification

Example
-------
python semantic_classify.py \
    --ply   /mnt/data/gaussianplant_data/pretrain_clean/newplant9_clean_pruned.ply \
    --root_path /mnt/data/gaussianplant_data \
    --branch_idx 1 --leaf_idx 0 --tau 0.07 --out output/semcls/newplant9
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def load_semantic_ply(path):
    """Read xyz + the semantic_* channels from a feature-3DGS ply."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    sem_names = sorted([p.name for p in v.properties if p.name.startswith("semantic_")],
                       key=lambda n: int(n.split("_")[1]))
    if not sem_names:
        raise ValueError(f"{path} has no semantic_* channels — is this a feature-3DGS ply?")
    sem = np.stack([v[n] for n in sem_names], axis=1).astype(np.float32)  # [N, D]
    return xyz, sem


def save_ply(path, xyz, extra_scalars=None, rgb=None):
    """Write a ply with optional per-point scalar fields and/or uint8 RGB."""
    cols = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    data = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    if extra_scalars:
        for name, arr in extra_scalars.items():
            cols.append((name, "f4")); data.append(arr.astype(np.float32))
    if rgb is not None:
        cols += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        data += [rgb[:, 0], rgb[:, 1], rgb[:, 2]]
    verts = np.empty(xyz.shape[0], dtype=[(n, t) for n, t in cols])
    for (n, _), d in zip(cols, data):
        verts[n] = d
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")]).write(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True, help="cleaned plant ply (with semantic_* channels)")
    ap.add_argument("--root_path", required=True, help="dir holding dinov3_text_feats.pth")
    ap.add_argument("--out", default="output/semcls", help="output directory")
    ap.add_argument("--branch_idx", type=int, default=1, help="text-prompt index used as BRANCH")
    ap.add_argument("--leaf_idx", type=int, default=0, help="text-prompt index used as LEAF")
    ap.add_argument("--tau", type=float, default=0.07, help="softmax temperature")
    ap.add_argument("--threshold", type=float, default=0.5, help="p_branch > threshold => branch")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    name = os.path.splitext(os.path.basename(args.ply))[0]

    # --- load features + text prompts ---
    xyz_np, sem_np = load_semantic_ply(args.ply)
    xyz = torch.from_numpy(xyz_np).to(dev)
    vf = F.normalize(torch.from_numpy(sem_np).to(dev), dim=-1)            # [N, D]

    tf = torch.load(os.path.join(args.root_path, "dinov3_text_feats.pth"),
                    map_location="cpu", weights_only=False)["text_feats_dim128"]
    tf = F.normalize(tf.to(dev), dim=-1)                                  # [P, D]
    cos = vf @ tf.T                                                       # [N, P]

    # --- report every prompt pair so the index convention is easy to pick ---
    P = tf.shape[0]
    print(f"[{name}] N={vf.shape[0]}  text prompts={P}  (branch_idx={args.branch_idx}, leaf_idx={args.leaf_idx})")
    print("  branch fraction if (branch,leaf) = ...")
    for b in range(P):
        row = []
        for l in range(P):
            if b == l:
                row.append("   -  ")
                continue
            p = F.softmax(torch.stack([cos[:, l], cos[:, b]], 1) / args.tau, 1)[:, 1]
            row.append(f"{(p > args.threshold).float().mean().item():5.2f}")
        print(f"    branch={b}: " + " ".join(row))

    # --- chosen classification ---
    logits = torch.stack([cos[:, args.leaf_idx], cos[:, args.branch_idx]], 1) / args.tau
    p_branch = F.softmax(logits, 1)[:, 1]                                 # [N] in [0,1]
    is_branch = p_branch > args.threshold
    print(f"  => branch {int(is_branch.sum())} / {is_branch.numel()} "
          f"({100 * is_branch.float().mean().item():.1f}%)")

    # --- write outputs ---
    p_np = p_branch.detach().cpu().numpy()
    lab_path = os.path.join(args.out, f"{name}_label.ply")
    save_ply(lab_path, xyz_np, extra_scalars={"label": p_np})

    rgb = np.zeros((xyz_np.shape[0], 3), np.uint8)
    b = is_branch.cpu().numpy()
    rgb[b] = np.array([220, 40, 40], np.uint8)      # branch = red
    rgb[~b] = np.array([40, 170, 60], np.uint8)     # leaf   = green
    col_path = os.path.join(args.out, f"{name}_color.ply")
    save_ply(col_path, xyz_np, rgb=rgb)

    # --- 3-view scatter png ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = np.random.RandomState(0).permutation(len(xyz_np))[:40000]
        fig = plt.figure(figsize=(13, 4.5))
        for k, (elev, azim) in enumerate([(12, -72), (12, 40), (85, -90)]):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            ax.scatter(xyz_np[s, 0], xyz_np[s, 1], xyz_np[s, 2], s=2, c=rgb[s] / 255.0)
            ax.view_init(elev, azim); ax.set_axis_off()
            c0 = xyz_np[s].mean(0); r = (xyz_np[s] - c0).ptp(0).max() / 2
            ax.set_xlim(c0[0]-r, c0[0]+r); ax.set_ylim(c0[1]-r, c0[1]+r); ax.set_zlim(c0[2]-r, c0[2]+r)
            ax.set_box_aspect((1, 1, 1))
        plt.suptitle(f"{name}: semantic branch(red)/leaf(green)  "
                     f"branch_idx={args.branch_idx} leaf_idx={args.leaf_idx} tau={args.tau}", fontsize=11)
        plt.tight_layout()
        png = os.path.join(args.out, f"{name}_semcls.png")
        plt.savefig(png, dpi=110, bbox_inches="tight", facecolor="white")
        print(f"  saved: {lab_path}\n         {col_path}\n         {png}")
    except Exception as e:
        print(f"  saved: {lab_path}\n         {col_path}\n  (png skipped: {e})")


if __name__ == "__main__":
    main()
