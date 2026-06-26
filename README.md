# GaussianPlant: Structure-Aligned Gaussian Splatting for Plant Structure Extraction
Yang Yang, Fumio Okura<br>
| [Webpage](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [Full Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) | <br>
This repository contains the official authors implementation associated with the paper "GaussianPlant: Structure-Aligned Gaussian Splatting for Plant Structure Extraction", which can be found [here](https://github.com/Yrainy0615/GaussianPlant.git). <br>
<a href="http://cvl.ist.osaka-u.ac.jp/en/"><img height="100" src="assets/osaka_logo.png"> </a>


Abstract: *We present a method for jointly recovering the appearance and structural organization of plants from multi-view images using 3D Gaussian Splatting (3DGS). While existing 3DGS approaches prioritize appearance fidelity or surface-aligned representations, they do not explicitly capture the underlying structure of objects. Our method introduces structural primitives (StPrs), initialized from clustered SfM points and represented as 3D Gaussians. These StPrs are first optimized to capture the coarse structural form of the plant, after which appearance Gaussians (AppGS) are bound to StPrs and jointly optimized through our proposed optimization strategy. The final plant structure is extracted from the optimized StPrs.
Our approach enables structure-aware 3DGS without requiring predefined skeleton priors or parametric templates. Experimental results demonstrate that our method effectively reconstructs both appearance and structure of plants, highlighting the potential of 3DGS as a framework for structural information extraction beyond scene representation.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>


## Cloning the Repository

The repository contains submodules, thus please check it out with
```shell
git clone git@github.com:Yrainy0615/GaussianPlant.git --recursive
```

## Installation

Tested on Ubuntu 22.04 with an NVIDIA RTX A6000 (driver CUDA 12.4, `nvcc` 12.4).
The rendering backend is [gsplat](https://github.com/nerfstudio-project/gsplat)
(the original `diff-gaussian-rasterization` is no longer required), and PyTorch3D
is replaced by a small shim in `utils/pytorch3d_compat.py`, so the only CUDA
extensions that get compiled are gsplat (JIT, on first run) and `simple-knn`.

```shell
# 1. create the environment
conda create -y -n gsplant python=3.10
conda activate gsplant

# 2. PyTorch matching the system CUDA (12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Python dependencies
pip install -r requirements.txt

# 4. gsplat (rasterizer) and faiss (GPU k-means for StrPr init)
pip install gsplat faiss-gpu-cu12 ninja
pip install "numpy<2"          # faiss pulls numpy>=2; the rest of the code needs <2

# 5. simple-knn CUDA extension (build isolation off so it sees torch)
pip install --no-build-isolation ./submodules/simple-knn
```

**Step 1 only (feature 3DGS pretraining).** Step 2 (structure extraction, the rest of
this README) runs entirely on gsplat and needs nothing above. Step 1 uses the
upstream [Feature-3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs) project,
included as a submodule at `third_party/feature-3dgs`. It ships its **own** conda
environment (`feature_3dgs`, python 3.8 / cu118) and CUDA rasterizer — keep it
entirely separate from the `gsplant` env. Pull and set it up only if you intend to
(re)run Step 1:
```shell
git submodule update --init --recursive third_party/feature-3dgs
# then follow third_party/feature-3dgs/README.md (its own env + rasterizer build)
```

gsplat compiles its CUDA kernels the first time it is used (~90 s). For that it
needs `nvcc` and `ninja` on `PATH`. If compilation fails (e.g. *"Ninja is required
to load C++ extensions"*) — common in non-login shells, `nohup`, or cron — export
the toolchain explicitly before running:
```shell
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Quick check:
```shell
python -c "import torch, gsplat, faiss; from simple_knn._C import distCUDA2; \
print('torch', torch.__version__, 'gsplat', gsplat.__version__, 'cuda', torch.cuda.is_available())"
```

## Data layout

Datasets live under `--root_path` (e.g. `/mnt/data/gaussianplant_data`). A scene
(e.g. `newplant9`) is a COLMAP capture plus a pre-trained **feature 3DGS**:

```
newplant9/
├── sparse/0, images/, masks/, depths/   # COLMAP scene
├── dinov3_dim128/                        # per-view DINOv3+JaFAR feature maps, PCA-128 (Step 1a)
└── feature_pretrain/point_cloud/iteration_30000/
    ├── point_cloud.ply                   # 3DGS + 128-d semantic + label (StrPr/AppGS source)
    └── point_cloud_branch_dense.ply      # GT dense branch points (Chamfer target)
pretrain_clean/
└── newplant9_clean_pruned.ply            # background-removed feature cloud (StrPr/AppGS source)
```
Shared assets `dinov3_pca.pth` and `dinov3_text_feats.pth` sit at `--root_path`.

## Pipeline

```
  images ─▶ [Step 1a] DINOv3 + JaFAR    ─▶ [Step 1b] Feature-3DGS   ─▶ [Step 2] Structure extraction
            per-view feature maps          distill into feature 3DGS    (this repo: StrPr + AppGS)
            (preprocessing, dinov3_dim128) (submodule; feature_pretrain  branch cylinders, skeleton
                                            + pretrain_clean/*.ply)      (MST), bound AppGS
```

### Step 1 — Feature 3DGS pretraining

**Step 1a — Per-view features (data preprocessing; recommended, no code in this repo).**
The 2D semantic features that Step 1b distills are produced as an offline
preprocessing pass, *not* by Feature-3DGS's built-in LSeg/SAM encoders. We recommend:
extract patch features with a **DINO / DINOv3** backbone, then **upsample them to full
image resolution with [JaFAR](https://github.com/PaulCouairon/JaFAR)** (a learned
feature upsampler) to get dense, edge-aligned per-pixel feature maps; PCA-reduce to
**128-d** and write one map per view under `<scene>/dinov3_dim128/` (with the shared
`dinov3_pca.pth` at `--root_path`). These maps are the input to Step 1b — keep this as
your own data-preprocessing script; this repo intentionally does not ship it.

**Step 1b — Distillation into a feature 3DGS** *(via the Feature-3DGS submodule).*
Distillation is handled by the upstream
**[Feature-3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs)**, pinned as a
submodule at [`third_party/feature-3dgs`](third_party/feature-3dgs) (we no longer
maintain a custom pretraining script). Point it at the precomputed feature maps from
Step 1a (in place of its default encoder), set the feature dimension to **128**
(`NUM_SEMANTIC_CHANNELS` in
`third_party/feature-3dgs/.../diff-gaussian-rasterization-feature/cuda_rasterizer/config.h`),
and run it in its **own** environment:

```shell
git submodule update --init --recursive third_party/feature-3dgs
cd third_party/feature-3dgs   # create its env + build its rasterizer (see its README)
python train.py -s <scene> -m <scene>/feature_pretrain --speedup --iterations 30000
```

Step 2 only needs these produced assets (its input contract):
- `feature_pretrain/point_cloud/iteration_30000/point_cloud.ply` — 3DGS with the
  128-d semantic feature and a per-Gaussian branch/leaf label channel.
- `pretrain_clean/<scene>_clean_pruned.ply` — the same cloud with pot/background
  removed (what Step 2 clusters into StrPr).
- `dinov3_pca.pth`, `dinov3_text_feats.pth` at `--root_path`.

> Step 2 does **not** need Feature-3DGS, its env, or its rasterizer — it only reads
> the produced `.ply` assets above and runs on gsplat.

### Step 2 — Structure extraction (this repo)

Recommended full command (the automatic pipeline: clean cloud → joint StrPr init →
auto branch fraction → binding + axis alignment + graph + isolation pruning →
branch-only densification):

```shell
python train.py \
  --source_path /mnt/data/gaussianplant_data/newplant9 \
  --root_path   /mnt/data/gaussianplant_data \
  --model_path  output/newplant9/run \
  --clean_ply   pretrain_clean/newplant9_clean_pruned.ply \
  --label_init joint --branch_frac -1 --cluster_size 40 \
  --reg_bind \
  --reg_axis  --lambda_axis 1.0 \
  --reg_graph --graph_from 1000 --graph_interval 50 \
  --prune_isolated --prune_from 1500 --prune_interval 500 --prune_until 4000 \
  --densify_branch --branch_split_ratio 1.5 --max_strpr_num 4000 \
  --label_lr 0 --iterations 7000 --disable_viewer
```

> **Do not add `--reg_overlap`.** The SuGaR-style overlap regulariser minimises
> neighbour overlap by collapsing each Gaussian's cross-section, which streaks the
> **leaf** StrPr into thin needles (in-plane elongation went 1.8 → ~185 on some
> scenes) while leaving the branch Chamfer unchanged. It is off by default; keep it
> off unless you add a per-class variant that spares the leaf disks.

`--clean_ply` loads the background-removed feature cloud; StrPr are built by
clustering it and AppGS are bound to them. (Alternatively, drop `--clean_ply` and
pass `--load_iteration 30000 --rm_bg` to load the raw `feature_pretrain` and remove
the background on-the-fly via the 2D plant masks.)

Outputs in `output/<scene>/run/point_cloud/iteration_7000/`:
`strpr.ply` (all StrPr), `strpr_branch.ply` (branch StrPr), `appgas.ply`,
`branch.ply` (bound branch AppGS points), `branch_cylinder.ply`, `mst.ply`.

### Parameters that need per-scene tuning

Most defaults generalise, but these are the knobs to adjust if a scene looks wrong
(thresholds are deliberately exposed because point density / branch fraction vary
per plant):

| flag | default | what it controls / when to change |
|------|---------|-----------------------------------|
| `--branch_frac` | `-1` (auto) | Fraction of StrPr initialised as **branch**. `-1` auto-calibrates via Otsu (capped at 0.15). **This is the one knob that does not always auto-generalise:** on scenes where branches are a small minority Otsu over-estimates and hits the cap, labelling leaves as branch (symptom: high `pred2gt`/`ctr2gt`, false branch cylinders on foliage). Then set it manually — typically `0.03`–`0.18` depending on how branchy the plant is (e.g. newplant5 needs ≈0.05, not the auto 0.15). |
| `--cluster_size` | `40` | Avg points per StrPr cluster → StrPr **count/granularity**. Smaller = more, finer StrPr (denser scenes can go lower). |
| `--prune_iso_factor` | `2.5` | Isolation pruning: demote a branch StrPr whose distance to its k-th nearest branch exceeds `median × factor`. **Lower → prune floaters more aggressively** (raise if real thin branches get removed). |
| `--branch_split_ratio` | `1.5` | Branch densification trigger: split a branch StrPr when its bound AppGS spill `> ratio × cylinder length` along the axis. Lower → split sooner / finer skeleton. |
| `--densify_quantile` | `0.9` | *(only with the generic `--densify`)* densify the top `(1-q)` fraction by gradient. gsplat gradients are ~1e3× smaller than the original rasteriser, so this is a **quantile**, not a fixed threshold. |
| `--prune_green_z` | `0.8` | *(with `--prune_green`)* greenness z-score above which a floating StrPr is treated as a green leaf-float and removed. |
| `--lambda_axis` | `1.0` | Strength of the branch major-axis → branch-tangent alignment. |

Notes:
- `--label_lr 0` **freezes** the branch/leaf labels at their (joint-init) values —
  recommended, since the init classification (AUC ≈ 0.97) is reliable and letting
  the labels drift degrades the branch skeleton.
- Prefer `--densify_branch` (binding-driven, branch-only) over the generic
  `--densify`: the latter follows photometric gradient and densifies **leaves**, not
  the branch structure.

## Evaluation (Chamfer distance)

The primary metric is the Chamfer distance between the recovered branch structure
and `point_cloud_branch_dense.ply` — for both the StrPr cylinder mesh and the
bound AppGS branch points:
```shell
python eval_chamfer.py \
  --gt /mnt/data/gaussianplant_data/newplant9/feature_pretrain/point_cloud/iteration_30000/point_cloud_branch_dense.ply \
  --scan output/newplant9          # ranks every run, prints the best
```


