#
# GaussianPlant renderer — gsplat backend.
#
# Replaces the original diff-gaussian-rasterization (+ the custom feature fork).
# Two rasterization passes are used: one for RGB (spherical-harmonics, view
# dependent) and one for the per-Gaussian semantic feature (raw N-D channels).
#
import math

import torch
import torch.nn.functional as F
from gsplat import rasterization

from scene.gaussian_model import GaussianModel


# --------------------------------------------------------------------------- #
#  camera helpers                                                             #
# --------------------------------------------------------------------------- #
def _view_and_K(viewpoint_camera):
    """Return (viewmats [1,4,4] world->cam, Ks [1,3,3]) for gsplat.

    The 3DGS ``world_view_transform`` is the transpose of the world->cam matrix
    (it is stored column-major for the CUDA rasterizer), so we transpose it back.
    """
    device = viewpoint_camera.world_view_transform.device
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)  # [4,4] W2C
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    fx = W / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    fy = H / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))
    K = torch.tensor(
        [[fx, 0.0, W * 0.5], [0.0, fy, H * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )
    return viewmat[None], K[None], W, H


def _radii_to_scalar(radii):
    """gsplat radii may be [C,N] or [C,N,2]; collapse to per-Gaussian [N]."""
    r = radii
    if r.dim() == 3:            # [C, N, 2] -> per-axis, take max
        r = r.amax(dim=-1)
    return r.squeeze(0)         # [N]


def _rasterize(viewpoint_camera, means, quats, scales, opacities, colors,
               bg, sh_degree, render_mode):
    viewmats, Ks, W, H = _view_and_K(viewpoint_camera)
    out, _alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=sh_degree,
        render_mode=render_mode,
        backgrounds=bg[None] if bg is not None else None,
        packed=False,
        absgrad=False,
    )
    return out, info


# --------------------------------------------------------------------------- #
#  core render                                                                #
# --------------------------------------------------------------------------- #
def _prepare(pc, override_color):
    means = pc.get_xyz
    quats = pc.get_rotation                     # [N,4] wxyz, normalized
    scales = pc.get_scaling                     # [N,3]
    opacities = pc.get_opacity.squeeze(-1)      # [N]
    if override_color is None:
        colors = pc.get_features                # [N, K, 3] SH coeffs
        sh_degree = pc.active_sh_degree
    else:
        colors = override_color
        sh_degree = None
    return means, quats, scales, opacities, colors, sh_degree


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None):
    """Render RGB + depth + semantic feature map. Returns the standard dict."""
    means, quats, scales, opacities, colors, sh_degree = _prepare(pc, override_color)

    # RGB + expected depth
    out, info = _rasterize(viewpoint_camera, means, quats, scales, opacities,
                           colors, bg_color, sh_degree, render_mode="RGB+ED")
    rgb = out[0, ..., :3].permute(2, 0, 1)      # [3,H,W]
    depth = out[0, ..., 3:4].permute(2, 0, 1)   # [1,H,W]

    means2d = info["means2d"]                   # [1,N,2]
    if means2d.requires_grad:
        means2d.retain_grad()
    radii = _radii_to_scalar(info["radii"])     # [N]

    # semantic feature map (raw channels, no SH)
    feature_map = None
    sem = pc.get_semantic_feature
    if sem is not None and sem.numel() > 0:
        feats = sem.squeeze(1)                   # [N, D]
        fbg = torch.zeros(feats.shape[1], device=feats.device)
        fout, _ = _rasterize(viewpoint_camera, means, quats, scales, opacities,
                             feats, fbg, sh_degree=None, render_mode="RGB")
        feature_map = fout[0].permute(2, 0, 1)   # [D,H,W]

    return {
        "render": rgb,
        "viewspace_points": means2d,
        "visibility_filter": radii > 0,
        "radii": radii,
        "feature_map": feature_map,
        "depth": depth,
    }


def _render_with_opacity(viewpoint_camera, pc, means, quats, scales, opacities,
                         colors, bg_color, sh_degree):
    out, _ = _rasterize(viewpoint_camera, means, quats, scales, opacities,
                        colors, bg_color, sh_degree, render_mode="RGB")
    return out[0, ..., :3].permute(2, 0, 1)


def render_gsplant(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                   scaling_modifier=1.0, override_color=None,
                   render_only_leaf=False, render_only_branch=False):
    """Full render plus optional branch-only / leaf-only RGB passes.

    Branch/leaf isolation is done by suppressing the opacity of the other class
    (mirrors the original implementation, which the binding/graph losses rely on).
    """
    pkg = render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)

    rendered_leaf = None
    rendered_branch = None
    if render_only_leaf or render_only_branch:
        means, quats, scales, opacities, colors, sh_degree = _prepare(pc, override_color)
        branch_mask, leaf_mask, _ = pc.get_cls_mask(mode='geo')
        if render_only_leaf:
            op = torch.full_like(opacities, 1e-6)
            op[leaf_mask] = opacities[leaf_mask]
            rendered_leaf = _render_with_opacity(viewpoint_camera, pc, means, quats,
                                                 scales, op, colors, bg_color, sh_degree)
        if render_only_branch:
            op = torch.full_like(opacities, 1e-6)
            op[branch_mask] = opacities[branch_mask]
            rendered_branch = _render_with_opacity(viewpoint_camera, pc, means, quats,
                                                   scales, op, colors, bg_color, sh_degree)
    return pkg, rendered_leaf, rendered_branch


# --------------------------------------------------------------------------- #
#  semantic / editing helpers (used by render.py inference)                   #
# --------------------------------------------------------------------------- #
def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0]):
    features = F.normalize(features.float(), p=2, dim=1)
    scores = features @ query_features.T  # (N, n_texts)
    if scores.shape[-1] == 1:
        scores = scores[:, 0]
        scores = (scores >= score_threshold).float()
    else:
        scores = torch.softmax(scores, dim=-1)
        if score_threshold is not None:
            scores = scores[:, positive_ids].sum(-1)
            scores = (scores >= score_threshold).float()
        else:
            scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)
            scores = torch.isin(torch.argmax(scores, dim=-1),
                                torch.tensor(positive_ids, device=scores.device)).float()
    return scores


def render_edit(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                text_feature: torch.Tensor, edit_dict: dict,
                scaling_modifier=1.0, override_color=None):
    """Text-guided editing render (deletion / extraction / recolor)."""
    means, quats, scales, opacities, colors, sh_degree = _prepare(pc, override_color)
    opacities = opacities.clone()

    semantic_feature = pc.get_semantic_feature  # [N,1,D]
    positive_ids = edit_dict["positive_ids"]
    score_threshold = edit_dict["score_threshold"]
    op_dict = edit_dict["operations"]

    if "deletion" in op_dict:
        scores = calculate_selection_score(semantic_feature[:, 0, :], text_feature,
                                           score_threshold, positive_ids)
        opacities.masked_fill_(scores > 0.5, 0)
    if "extraction" in op_dict:
        scores = calculate_selection_score(semantic_feature[:, 0, :], text_feature,
                                           score_threshold, positive_ids)
        opacities.masked_fill_(scores < 0.5, 0)
    if "color_func" in op_dict and sh_degree is not None:
        scores = calculate_selection_score(semantic_feature[:, 0, :], text_feature,
                                           score_threshold, positive_ids)
        colors = colors.clone()
        colors[:, 0, :] = colors[:, 0, :] * (1 - scores[:, None]) + \
            op_dict["color_func"](colors[:, 0, :]) * scores[:, None]

    out, info = _rasterize(viewpoint_camera, means, quats, scales, opacities,
                           colors, bg_color, sh_degree, render_mode="RGB+ED")
    radii = _radii_to_scalar(info["radii"])
    return {
        "render": out[0, ..., :3].permute(2, 0, 1),
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": out[0, ..., 3:4].permute(2, 0, 1),
    }


def render_dynamic(viewpoint_camera, pc: GaussianModel, pipe, nn_stpr_app, bg_color,
                   stpr_label, scaling_modifier=1.0, override_color=None):
    """Render each leaf instance separately (one StrPr at a time). Returns a list."""
    means, quats, scales, opacities, colors, sh_degree = _prepare(pc, override_color)
    parent_id = nn_stpr_app.squeeze(1)
    leaf_index = torch.nonzero(torch.sigmoid(stpr_label) >= 0.5, as_tuple=False)
    leaf_index = leaf_index[:, 0] if leaf_index.dim() > 1 else leaf_index

    image_list = []
    for i in torch.unique(parent_id):
        if i in leaf_index:
            op = opacities.clone()
            op[parent_id == i] = 1e-6
            img = _render_with_opacity(viewpoint_camera, pc, means, quats, scales,
                                       op, colors, bg_color, sh_degree).clamp(0, 1)
            image_list.append(img)
    return image_list
