<div align="center">

# 🌱 GaussianPlant

### Structure-aligned Gaussian Splatting for 3D Reconstruction of Plants

**Yang Yang · Risa Shinoda · Hiroaki Santo · Fumio Okura**

[![arXiv](https://img.shields.io/badge/arXiv-2512.14087-b31b1b.svg)](https://arxiv.org/abs/2512.14087)
[![GitHub](https://img.shields.io/badge/GitHub-GaussianPlant-181717.svg?logo=github)](https://github.com/Yrainy0615/GaussianPlant)

<a href="http://cvl.ist.osaka-u.ac.jp/en/"><img height="64" src="assets/osaka_logo.png"></a>

<img src="configs/teaser_gsplant.svg" width="100%">

</div>

> **GaussianPlant** jointly recovers a plant's **appearance** and its **structure** from
> multi-view images using 3D Gaussian Splatting. Structure primitives (StPrs) model
> branches as cylinders and leaves as disks; appearance Gaussians (AppGS) are bound to
> them and jointly optimized through a re-rendering loss. The output is a high-fidelity
> render **plus** an explicit branch skeleton and per-leaf instances — without predefined
> skeleton priors or parametric templates.

---

## Contents

- [Installation](#installation)
- [Data layout](#data-layout)
- [Pipeline](#pipeline)
  - [Step 1 — Feature 3DGS pretraining](#step-1--feature-3dgs-pretraining)
  - [Step 2 — Structure extraction](#step-2--structure-extraction-this-repo)
- [Parameters to tune](#parameters-to-tune)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## Installation

Tested on Ubuntu 22.04 with an NVIDIA RTX A6000 (CUDA 12.4, `nvcc` 12.4). The rendering
backend is [gsplat](https://github.com/nerfstudio-project/gsplat) (the original
`diff-gaussian-rasterization` is no longer required); PyTorch3D is replaced by a small
shim in `utils/pytorch3d_compat.py`. The only CUDA extensions compiled are gsplat (JIT,
on first run) and `simple-knn`.

**1. Clone (with submodules)**

```shell
git clone https://github.com/Yrainy0615/GaussianPlant.git --recursive
cd GaussianPlant
```

**2. Environment**

```shell
conda create -y -n gsplant python=3.10
conda activate gsplant

# PyTorch matching the system CUDA (12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install gsplat faiss-gpu-cu12 ninja
pip install "numpy<2"          # faiss pulls numpy>=2; the rest of the code needs <2
pip install --no-build-isolation ./submodules/simple-knn
```

**3. Verify**

```shell
python -c "import torch, gsplat, faiss; from simple_knn._C import distCUDA2; \
print('torch', torch.__version__, 'gsplat', gsplat.__version__, 'cuda', torch.cuda.is_available())"
```

> [!NOTE]
> gsplat compiles its kernels on first use (~90 s) and needs `nvcc` + `ninja` on `PATH`.
> If it fails with *"Ninja is required to load C++ extensions"* (common in non-login
> shells / `nohup` / cron), export the toolchain first:
> ```shell
> export CUDA_HOME=/usr/local/cuda-12.4
> export PATH=$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH
> export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
> ```

> [!IMPORTANT]
> The above is all **Step 2** (this repo) needs. **Step 1** (feature-3DGS pretraining)
> uses the [Feature-3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs) submodule,
> which has its **own** conda env (`feature_3dgs`, py3.8 / cu118) and rasterizer — set it
> up separately, and only if you intend to (re)run Step 1.

---

## Data layout

Datasets live under `--root_path` (e.g. `/mnt/data/gaussianplant_data`). A scene is a
COLMAP capture plus a pretrained **feature 3DGS**:

```
<root_path>/
├── dinov3_pca.pth, dinov3_text_feats.pth      # shared assets
├── pretrain_clean/
│   └── <scene>_clean_pruned.ply               # plant-only feature cloud — PRODUCED, see note (StrPr/AppGS source)
└── <scene>/
    ├── sparse/0, images/, masks/, depths/     # COLMAP scene
    ├── dinov3_dim128/                         # per-view DINOv3+JaFAR feature maps, PCA-128 (Step 1a)
    └── feature_pretrain/point_cloud/iteration_30000/
        ├── point_cloud.ply                    # 3DGS + 128-d semantic feature (no label; see note)
        └── point_cloud_branch_dense.ply       # GT dense branch points — EVALUATION ONLY (not used in training)
```

> [!NOTE]
> - **`pretrain_clean/*_clean_pruned.ply` is not a given asset** — it is the feature
>   cloud with the pot/background removed, which you produce yourself: either by
>   **DBSCAN** on the feature cloud (keep the plant cluster) or by projecting the 2D
>   plant **masks** to 3D (the built-in `--rm_bg` path, `compute_plant_mask`). Pass the
>   result via `--clean_ply`, or skip it and let `--rm_bg` build it on the fly.
> - **The feature 3DGS carries no branch/leaf label.** `point_cloud.ply` stores colour
>   + the 128-d semantic feature only; the branch/leaf label is **derived in Step 2** at
>   init (`build_strpr_from_gs`, joint = colour + geometry + semantic), not loaded from
>   the pretrain.
> - **Training uses no ground truth.** Step 2 is fully self-supervised (re-rendering
>   loss against the images + feature pretrain). `point_cloud_branch_dense.ply` is read
>   **only** by `eval_chamfer.py` to score the result.
> - **The semantic assets ship in [`assets/`](assets/)** so the code works out of the box:
>   `dinov3_text_feats.pth` (16 KB, branch/leaf/bg text prompts) and `dinov3_pca.pth`
>   (3.4 MB, the 1024→768→128 PCA basis). The PCA basis is **required for Step-1a** to
>   project new features into the same 128-d space as the text prompts (see Step 1a).

---

## Pipeline

```
 images ─▶ [1a] DINOv3 + JaFAR    ─▶ [1b] Feature-3DGS      ─▶ [2] Structure extraction
           per-view feature maps      distill → feature 3DGS    (this repo: StrPr + AppGS)
           (preprocessing)            (submodule)               branch skeleton + leaf instances
```

### Step 1 — Feature 3DGS pretraining

**1a · Per-view features** — *data preprocessing, no code in this repo.*
The 2D semantic features distilled in Step 1b are produced offline (not by Feature-3DGS's
built-in LSeg/SAM encoders). Recommended: extract patch features with a **DINO / DINOv3**
backbone, upsample to full resolution with **[JaFAR](https://github.com/PaulCouairon/JaFAR)**
(a learned feature upsampler) for dense, edge-aligned per-pixel maps, project to **128-d**,
and write one map per view under `<scene>/dinov3_dim128/`. Keep this as your own
preprocessing script — this repo intentionally does not ship it.

> [!IMPORTANT]
> **Use the provided PCA basis for the 128-d projection — don't fit your own.** The
> branch/leaf **text** features shipped in [`assets/`](assets/) were reduced to 128-d with
> the exact basis in [`assets/dinov3_pca.pth`](assets/dinov3_pca.pth) (`pca`: 1024→768,
> `pca128`: 768→128). Your per-view visual features must be projected with the **same**
> basis, otherwise the visual and text features live in different 128-d subspaces and the
> semantic cue (`get_semantic_prior`, the joint-init semantic term) is meaningless. Apply
> it as `feat128 = pca128.transform(pca.transform(feat1024))` (or just `pca128.transform`
> if your backbone already outputs 768-d). Verified: `pca128.transform(text_768)` reproduces
> the shipped `text_feats_dim128` exactly.

**1b · Distillation** — *via the Feature-3DGS submodule.*
Handled by upstream [Feature-3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs) at
`third_party/feature-3dgs`. Point it at the Step-1a feature maps (in place of its default
encoder) and set the feature dimension to **128** (`NUM_SEMANTIC_CHANNELS` in its
`.../diff-gaussian-rasterization-feature/cuda_rasterizer/config.h`):

```shell
git submodule update --init --recursive third_party/feature-3dgs
cd third_party/feature-3dgs        # create its env + build its rasterizer (see its README)
python train.py -s <scene> -m <scene>/feature_pretrain --speedup --iterations 30000
```

**Step-2 input contract** — whatever the encoder, Step 2 only consumes:
| asset | description |
|-------|-------------|
| `feature_pretrain/.../point_cloud.ply` | 3DGS with the 128-d semantic feature (no label channel) |
| `pretrain_clean/<scene>_clean_pruned.ply` | the same cloud, pot/background removed |
| `dinov3_pca.pth`, `dinov3_text_feats.pth` | shared assets at `--root_path` |

### Step 2 — Structure extraction (this repo)

Automatic pipeline: clean cloud → joint StrPr init → auto branch fraction → binding +
axis alignment + graph + isolation pruning → branch-only densification.

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

`--clean_ply` loads the background-removed cloud; StrPr are clustered from it and AppGS
bound to them. (Alternatively, drop `--clean_ply` and pass `--load_iteration 30000 --rm_bg`
to load the raw `feature_pretrain` and mask the background on the fly.)

**Outputs** in `output/<scene>/run/point_cloud/iteration_7000/`:

| file | content |
|------|---------|
| `strpr.ply` | all StrPr (full Gaussians) |
| `strpr_branch.ply` | branch StrPr only |
| `branch.ply` | bound branch AppGS points (Chamfer prediction) |
| `mst.ply` | branch MST skeleton |
| `appgas.ply` | appearance Gaussians |
| `point_cloud.ply` | StrPr + AppGS merged (any 3DGS viewer) |

> [!WARNING]
> **Do not add `--reg_overlap`.** The SuGaR-style overlap regulariser minimises neighbour
> overlap by collapsing each Gaussian's cross-section, which streaks the **leaf** StrPr
> into thin needles (in-plane elongation 1.8 → ~185 on some scenes) with no benefit to the
> branch Chamfer. It is off by default — keep it off.

---

## Parameters to tune

Most defaults generalise; these are the knobs to adjust per scene (point density and
branch fraction vary by plant):

| flag | default | what it controls / when to change |
|------|:-------:|-----------------------------------|
| `--branch_frac` | `-1` (auto) | Fraction of StrPr labelled **branch**. `-1` auto-calibrates via Otsu (capped 0.15). **The one knob that doesn't always auto-generalise:** when branches are a small minority, Otsu over-estimates and hits the cap, labelling leaves as branch (symptom: high `pred2gt`/`ctr2gt`, false cylinders on foliage). Set manually `0.03`–`0.18` (e.g. newplant5 ≈ 0.05). |
| `--cluster_size` | `40` | Avg points per StrPr cluster → StrPr **count / granularity**. Smaller = more, finer StrPr. |
| `--prune_iso_factor` | `2.5` | Isolation pruning: demote a branch StrPr whose distance to its k-th nearest branch > `median × factor`. Lower → prune floaters harder. |
| `--branch_split_ratio` | `1.5` | Branch densification: split a branch StrPr when its bound AppGS spill `> ratio × cylinder length`. Lower → finer skeleton. |
| `--prune_green_z` | `0.8` | *(with `--prune_green`)* greenness z-score above which a floating StrPr is removed as a green leaf-float. |
| `--lambda_axis` | `1.0` | Strength of the branch major-axis → branch-tangent alignment. |
| `--densify_quantile` | `0.9` | *(only with generic `--densify`)* densify the top `(1-q)` fraction by gradient (gsplat grads are ~1e3× smaller, so a quantile not a fixed threshold). |

**Recommended defaults**
- `--label_lr 0` **freezes** the branch/leaf labels at their joint-init values (init AUC ≈ 0.97; letting them drift degrades the skeleton).
- Prefer `--densify_branch` (binding-driven, branch-only) over generic `--densify` (which follows photometric gradient and densifies **leaves**, not branches).

---

## Optional: semantic-only branch/leaf labelling

`semantic_classify.py` is a small **standalone** helper that labels a *cleaned* cloud
branch/leaf purely from the **DINOv3 text similarity** of each point's 128-d feature —
**no colour, no geometry**. It exists for the case where the colour cue in the joint init
fails (e.g. **yellow / autumn leaves**, or branch and leaf colours that are hard to tell
apart).

```shell
python semantic_classify.py \
  --ply /mnt/data/gaussianplant_data/pretrain_clean/<scene>_clean_pruned.ply \
  --root_path /mnt/data/gaussianplant_data \
  --branch_idx 1 --leaf_idx 0 --out output/semcls/<scene>
```
It writes a `*_label.ply` (branch probability), a red/green `*_color.ply`, and a 3-view
`*_semcls.png`, and prints the branch fraction for every text-prompt pair (the prompt
convention is undocumented, so try a few `--branch_idx/--leaf_idx`).

> [!WARNING]
> This is a **niche fallback, not the default path.** The pure text-similarity signal is
> soft (per-point AUC ≈ 0.6–0.85, scene/prompt dependent). Use it **only on structurally
> simple plants with almost no self-occlusion** — occlusion makes the per-point feature
> unreliable. For normal scenes the joint init (colour + geometry + semantic) is far more
> accurate; if only colour is the problem, prefer fixing the colour cue over this.

---

## Evaluation

Primary metric: Chamfer distance between the recovered branch points and the GT dense
branch cloud `point_cloud_branch_dense.ply`.

```shell
python eval_chamfer.py \
  --gt /mnt/data/gaussianplant_data/newplant9/feature_pretrain/point_cloud/iteration_30000/point_cloud_branch_dense.ply \
  --scan output/newplant9          # ranks every run, prints the best
```

---

## Citation

```bibtex
@article{yang2025gaussianplant,
  title   = {GaussianPlant: Structure-aligned Gaussian Splatting for 3D Reconstruction of Plants},
  author  = {Yang, Yang and Shinoda, Risa and Santo, Hiroaki and Okura, Fumio},
  journal = {arXiv preprint arXiv:2512.14087},
  year    = {2025}
}
```
