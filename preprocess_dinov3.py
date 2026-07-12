#!/usr/bin/env python
"""
Step-1a preprocessing: extract per-view DINOv3 semantic features + the ALIGNED
branch/leaf text features for a scene.

Why this script exists
----------------------
GaussianPlant's semantic branch/leaf cue works by comparing each Gaussian's DINOv3
feature to `branch` / `leaf` **text** features. For that comparison to mean anything,
the per-view visual features and the text features MUST live in the *same* reduced
space. The one reliable way to guarantee that is to **fit a single PCA on the visual
features and project the text features with that exact same PCA** — which is what this
script does. (A common failure we have seen: fitting a per-batch PCA on colour/mask-
augmented features, then comparing against text projected with a different PCA. The
visual and text features then sit in different subspaces, cosine(visual, text) collapses
to ~0, and the semantic cue is dead. Keep the semantic feature PURE DINOv3 here; carry
colour / masks as separate channels downstream, never mixed into the semantic vector.)

Outputs (DINOv3 ViT-L/16 dinotxt, patch-token features are 1024-d):
  <root>/<scene>/dinov3_dim128/<view>_dinov3_128.pth   # [128, H, W] per-view feature map
  <root>/dinov3_pca.pth                                 # {'pca': TorchPCA(1024->128)}
  <root>/dinov3_text_feats.pth                          # {'text_feats_dim128','text_feats_dim1024','prompts'}

Requires the DINOv3 dinotxt weights (see README) and internet for the BPE vocab on first
run. Feed the produced feature maps to Feature-3DGS (Step 1b) with NUM_SEMANTIC_CHANNELS=128.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.visualization import TorchPCA

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PATCH = 16


def load_dinotxt(ckpt_dir, device):
    repo = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.join(repo, "dinov3")
    model, tokenizer = torch.hub.load(
        repo, "dinov3_vitl16_dinotxt_tet1280d20h24l", source="local",
        dinotxt_weights=os.path.join(ckpt_dir, "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"),
        backbone_weights=os.path.join(ckpt_dir, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
    return model.to(device).eval(), tokenizer


def preprocess_image(path, long_side, device):
    """Load an image, resize (keep aspect, sides multiple of PATCH), ImageNet-normalise."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = long_side / max(w, h)
    nw, nh = max(PATCH, round(w * scale / PATCH) * PATCH), max(PATCH, round(h * scale / PATCH) * PATCH)
    img = img.resize((nw, nh), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1)[None] / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.to(device), (nh // PATCH, nw // PATCH)


@torch.no_grad()
def patch_features(model, img, grid):
    """Return the 1024-d patch-token feature grid [Hp, Wp, 1024] for one image."""
    _, patch_tokens, _ = model.encode_image_with_patch_tokens(img)   # [1, Np(+cls?), D]
    pt = patch_tokens[0]
    Hp, Wp = grid
    if pt.shape[0] == Hp * Wp + 1:      # drop a leading CLS/register token if present
        pt = pt[1:]
    assert pt.shape[0] == Hp * Wp, f"patch count {pt.shape[0]} != {Hp*Wp}"
    return pt.reshape(Hp, Wp, -1).float()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_path", required=True, help="dataset root (shared assets written here)")
    ap.add_argument("--scene", required=True, help="scene name under root_path")
    ap.add_argument("--images", default="images", help="image subdir")
    ap.add_argument("--mask_dir", default="", help="optional plant-mask subdir (fit PCA on foreground only)")
    ap.add_argument("--ckpt_dir", default="checkpoints", help="dir with the DINOv3 dinotxt weights")
    ap.add_argument("--long_side", type=int, default=768, help="resize long side (multiple of 16)")
    ap.add_argument("--out_dim", type=int, default=128)
    ap.add_argument("--pca_sample", type=int, default=300000, help="#patch vectors used to fit the PCA")
    # templated prompts align markedly better than bare words with DINOv3-dinotxt
    ap.add_argument("--prompts", nargs="+",
                    default=["a photo of a plant branch", "a photo of a green leaf",
                             "a photo of background", "a photo of a plant"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    scene_dir = os.path.join(args.root_path, args.scene)
    img_paths = sorted(sum([glob.glob(os.path.join(scene_dir, args.images, f"*.{e}"))
                            for e in ("jpg", "JPG", "png", "PNG", "jpeg")], []))
    assert img_paths, f"no images under {scene_dir}/{args.images}"
    out_dir = os.path.join(scene_dir, "dinov3_dim128"); os.makedirs(out_dir, exist_ok=True)
    print(f"{args.scene}: {len(img_paths)} views -> {out_dir}")

    model, tokenizer = load_dinotxt(args.ckpt_dir, dev)

    # ---- pass 1: cache patch grids, collect a sample of foreground vectors for the PCA ----
    grids, sample = {}, []
    for p in img_paths:
        img, grid = preprocess_image(p, args.long_side, dev)
        feat = patch_features(model, img, grid)                      # [Hp,Wp,1024]
        grids[p] = feat.cpu()
        v = feat.reshape(-1, feat.shape[-1])
        if args.mask_dir:
            mp = _find_mask(scene_dir, args.mask_dir, p)
            if mp:
                m = Image.open(mp).convert("L").resize((grid[1], grid[0]), Image.NEAREST)
                keep = torch.from_numpy(np.asarray(m) > 127).reshape(-1)
                v = v[keep]
        if v.shape[0]:
            sample.append(v[torch.randperm(v.shape[0])[:4000]])
    sample = torch.cat(sample, 0)
    idx = torch.randperm(sample.shape[0])[:args.pca_sample]
    print(f"fitting PCA 1024->{args.out_dim} on {len(idx)} patch vectors ...")
    pca = TorchPCA(args.out_dim).fit(sample[idx].to(dev))

    # ---- pass 2: project every view, save [128, H, W] (upsampled to image resolution) ----
    for p in img_paths:
        feat = grids[p].to(dev)                                      # [Hp,Wp,1024]
        Hp, Wp, _ = feat.shape
        red = pca.transform(feat.reshape(-1, feat.shape[-1])).reshape(Hp, Wp, args.out_dim)
        red = red.permute(2, 0, 1)                                  # [128,Hp,Wp] (patch-grid res)
        # save at the feature-grid resolution; the training loader interpolates to image size.
        name = os.path.splitext(os.path.basename(p))[0]
        torch.save(red.half().cpu(), os.path.join(out_dir, f"{name}_dinov3_128.pth"))
    print(f"saved {len(img_paths)} feature maps")

    # ---- text features: encode prompts, project with the SAME PCA (this is the alignment) ----
    t2048 = model.encode_text(tokenizer.tokenize(args.prompts).to(dev))
    t1024 = t2048[:, 1024:].float()                                 # patch-aligned half
    t128 = pca.transform(t1024)                                     # SAME pca as the visual feats
    # save under both keys ('pca' and 'pca128') for the Scene loader (single 1024->128 stage here)
    torch.save({"pca": pca, "pca128": pca}, os.path.join(args.root_path, "dinov3_pca.pth"))
    torch.save({"prompts": args.prompts,
                "text_feats_dim1024": t1024.cpu(),
                "text_feats_dim128": t128.cpu()},
               os.path.join(args.root_path, "dinov3_text_feats.pth"))

    # sanity: mean cosine(visual, text) should be clearly positive (~0.1-0.2), not ~0
    vis128 = pca.transform(sample[:20000].to(dev))                  # project visual with the SAME pca
    vs = F.normalize(vis128, dim=1) @ F.normalize(t128, dim=1).T
    print("alignment check — mean cos(patch feats, each prompt):", vs.mean(0).cpu().numpy().round(3),
          "  (all >~0.1 => visual & text share the space)")
    print("done. wrote dinov3_pca.pth + dinov3_text_feats.pth to", args.root_path)


def _find_mask(scene_dir, mask_dir, img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    for e in ("png", "jpg", "JPG", "PNG"):
        m = os.path.join(scene_dir, mask_dir, f"{base}.{e}")
        if os.path.exists(m):
            return m
    return None


if __name__ == "__main__":
    main()
