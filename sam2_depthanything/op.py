from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Bool, Float32, Float64, UInt8
from monopriors.depth_utils import clip_disparity, depth_edges_mask, depth_to_points
from monopriors.relative_depth_models.depth_anything_v2 import (
    RelativeDepthPrediction,
)


def log_relative_pred(
    parent_log_path: Path,
    relative_pred: RelativeDepthPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    seg_mask_hw: UInt8[np.ndarray, "h w"] | None = None,
    remove_flying_pixels: bool = True,
    jpeg_quality: int = 90,
    depth_edge_threshold: float = 1.1,
) -> None:
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    # assume camera is at the origin
    cam_T_world_44: Float64[np.ndarray, "4 4"] = np.eye(4)

    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=cam_T_world_44[:3, 3],
            mat3x3=cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rr.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=relative_pred.K_33,
            width=rgb_hw3.shape[1],
            height=rgb_hw3.shape[0],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_hw3).compress(jpeg_quality=jpeg_quality))

    depth_hw: Float32[np.ndarray, "h w"] = relative_pred.depth
    if remove_flying_pixels:
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=depth_edge_threshold)
        rr.log(
            f"{pinhole_path}/edge_mask",
            rr.SegmentationImage(edges_mask.astype(np.uint8)),
        )
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw * ~edges_mask

    if seg_mask_hw is not None:
        rr.log(
            f"{pinhole_path}/segmentation",
            rr.SegmentationImage(seg_mask_hw),
        )
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw  # * seg_mask_hw

    rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw))

    # removes outliers from disparity (sometimes we can get weirdly large values)
    clipped_disparity: UInt8[np.ndarray, "h w"] = clip_disparity(relative_pred.disparity)

    # log to cam_log_path to avoid backprojecting disparity
    rr.log(f"{cam_log_path}/disparity", rr.DepthImage(clipped_disparity))

    depth_1hw: Float32[np.ndarray, "1 h w"] = rearrange(depth_hw, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, relative_pred.K_33)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=rgb_hw3.reshape(-1, 3),
        ),
    )


def create_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    contents: list = [
        rrb.Vertical(
            rrb.Spatial2DView(
                origin=f"{pinhole_path}/image",
            ),
            rrb.Spatial2DView(
                origin=f"{pinhole_path}/segmentation",
            ),
            rrb.Spatial2DView(
                origin=f"{cam_log_path}/disparity",
            ),
        ),
        rrb.Spatial3DView(origin=f"{parent_log_path}"),
    ]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(contents=contents, column_shares=[1, 3]),
        collapse_panels=True,
    )
    return blueprint
