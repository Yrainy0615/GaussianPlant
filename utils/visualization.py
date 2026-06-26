import torch
import torch.nn as nn
import sklearn.decomposition
import numpy as np
from torch.nn import functional as F
import pyvista as pv

class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=True, niter=7)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected
    
def flatten(tensor):
    B, C, H, W = tensor.shape
    return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].detach().cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature

def feature_map_vis(feature):
    if feature.dim() == 3:
        feature = feature[None, ...]
    b,c,h,w = feature.shape
    feature = F.normalize(feature, p=2, dim=1)
    pca = TorchPCA(3)
    pca.fit(flatten(feature))
    x_red2 = pca.transform(flatten(feature))
    feature_vis =  x_red2.reshape(b,h,w,3).permute(0, 3, 1, 2).contiguous().float()
    feature_vis = F.normalize(feature_vis, p=2, dim=1)
    return feature_vis.squeeze(0)

# visualizer.py
import os
import time
import pathlib
from typing import Optional, Union



class Visualizer:
    """
    用 PyVista 可视化点云。
    - 按 's' 保存当前视角截图（文件名含时间戳）
    - 支持 numpy 数组或常见点云文件（.ply/.vtp/.obj 等，取其顶点作为点云）
    """

    def __init__(
        self,
        out_dir: Union[str, os.PathLike] = "captures",
        window_size: tuple[int, int] = (1280, 720),
        background: str = "black",
    ):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.background = background

    # -------- public API --------
    def show_point_cloud(
        self,
        data: Union[str, os.PathLike, np.ndarray],
        *,
        colors: Optional[np.ndarray] = None,
        color: Optional[str] = "white",
        point_size: float = 3.0,
        name: Optional[str] = None,
        eye_dome_lighting: bool = True,
        show_axes: bool = True,
    ) -> None:
        pts, rgb = self._to_points_and_colors(data, colors=colors)
        base = name or self._default_name_from_data(data)

        pl = pv.Plotter(window_size=self.window_size)
        pl.set_background(self.background)

        # 添加点云
        if rgb is not None:
            pl.add_points(
                pts,
                scalars=rgb,
                rgb=True,
                render_points_as_spheres=True,
                point_size=point_size,
            )
        else:
            pl.add_points(
                pts,
                color=color,
                render_points_as_spheres=True,
                point_size=point_size,
            )

        if eye_dome_lighting:
            pl.enable_eye_dome_lighting()  # 增强点云的立体感

        if show_axes:
            pl.add_axes(line_width=2)

        pl.show_bounds(grid='front', location='outer', all_edges=True)
        pl.camera_position = "xy"  # 先给个正交俯视作为起点，用户可自由旋转

        # 绑定保存截图的按键
        pl.add_key_event("s", self._make_save_callback(pl, base))

        # 进入交互（此时用户可以旋转/缩放，按 s 截图，按 q 退出）
        pl.show()

    # -------- internal helpers --------
    def _make_save_callback(self, plotter: pv.Plotter, base: str):
        def _save():
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self.out_dir / f"{base}_{ts}.png"
            plotter.screenshot(str(path))
            print(f"[Visualizer] Saved screenshot: {path}")
        return _save

    def _default_name_from_data(self, data) -> str:
        if isinstance(data, (str, os.PathLike)):
            stem = pathlib.Path(data).stem
            return stem if stem else "cloud"
        return "cloud"

    def _to_points_and_colors(
        self,
        data: Union[str, os.PathLike, np.ndarray],
        colors: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        返回 (N,3) 点坐标，以及可选 (N,3) uint8 RGB。
        """
        if isinstance(data, (str, os.PathLike)):
            mesh = pv.read(str(data))
            # 取顶点作为点云
            pts = np.asarray(mesh.points, dtype=np.float32)
            rgb = None

            # 若文件中自带颜色（常见于 PLY），尝试读取
            # PyVista 会把颜色放在 point_data 中可能叫 'RGBA' 或 'RGB' 或 'Colors'
            for key in ("RGBA", "RGB", "Colors", "colors", "Scalars_"):
                if key in mesh.point_data:
                    arr = np.asarray(mesh.point_data[key])
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        rgb = arr[:, :3].astype(np.uint8)
                        break

            # 外部传入 colors 优先
            if colors is not None:
                rgb = self._validate_colors(colors, pts.shape[0])

            return pts, rgb

        # numpy 数组
        arr = np.asarray(data)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expect (N,3) points, got shape {arr.shape}")
        pts = arr.astype(np.float32)

        rgb = None
        if colors is not None:
            rgb = self._validate_colors(colors, pts.shape[0])
        return pts, rgb

    def _validate_colors(self, colors: np.ndarray, n_pts: int) -> np.ndarray:
        c = np.asarray(colors)
        if c.shape != (n_pts, 3):
            raise ValueError(f"colors must be shape ({n_pts}, 3), got {c.shape}")
        if c.dtype != np.uint8:
            c = np.clip(c, 0, 255).astype(np.uint8)
        return c
import numpy as np
import pyvista as pv
from dataclasses import dataclass
from typing import Literal, Optional





@dataclass
class TubeStyle:
    color: str = "#5c9ef5"
    roughness: float = 0.8
    metallic: float = 0.0
    specular: float = 0.2
    opacity: float = 1.0
    n_sides: int = 30          # 管壁细分越高越圆
    capping: bool = True       # 封端
    background: str = "white"
    show_bounds: bool = False
    eye_dome_lighting: bool = True
    ambient: float = 0.2
    diffuse: float = 0.9


def make_tube_mesh_from_graph(
    poly_graph: pv.PolyData,
    *,
    radius_scalar_name: str = "radius",
    tube_style: Optional[TubeStyle] = None
) -> pv.PolyData:
    """
    把折线图转为“粗线管”网格。半径来自 point_data['radius']，在边上做线性插值。
    """
    tube_style = tube_style or TubeStyle()
    # absolute=True: 标量就是绝对半径（以 points 单位计），不是比例
    tube = poly_graph.tube(
        scalars=radius_scalar_name,
        n_sides=tube_style.n_sides,
        capping=tube_style.capping,
        absolute=True
    )
    # 可选：一点平滑（tube 已经是规则网格，通常不需要）
    # tube = tube.smooth(n_iter=10, relaxation_factor=0.1, feature_smoothing=False)
    return tube



if __name__ == "__main__":
    points = np.random.rand(10000, 3) * 2 - 1  # [-1,1] 范围随机点
    viz = Visualizer(out_dir="output/captures", window_size=(1400, 900))
    viz.show_point_cloud(points, point_size=4.0, color="cyan", name="random_cloud")

    # 2) 可视化带颜色的点云
    colors = (np.clip((points + 1) / 2 * 255, 0, 255)).astype(np.uint8)
    viz.show_point_cloud(points, colors=colors, point_size=3.0, name="colored_cloud")
