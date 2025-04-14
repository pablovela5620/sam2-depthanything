import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import cv2
import gradio as gr
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from gradio_rerun import Rerun
from gradio_rerun.events import (
    SelectionChange,
)
from icecream import ic
from jaxtyping import Bool, Float, Float32, Int, UInt8, UInt16
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from simplecv.camera_parameters import PinholeParameters
from simplecv.conversion_utils import save_to_nerfstudio
from simplecv.data.exoego.assembly_101 import Assembely101Sequence
from simplecv.data.exoego.hocap import ExoCameraIDs, HOCapSequence
from simplecv.ops.triangulate import batch_triangulate, projectN3
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.video_io import MultiVideoReader

if gr.NO_RELOAD:
    VIDEO_SAM_PREDICTOR: SAM2VideoPredictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    IMG_SAM_PREDICTOR: SAM2ImagePredictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")


def create_blueprint(exo_video_log_paths: list[Path], num_videos_to_log: Literal[4, 8] = 8) -> rrb.Blueprint:
    active_tab: int = 0  # 0 for video, 1 for images
    main_view = rrb.Vertical(
        contents=[
            rrb.Spatial3DView(
                origin="/",
            ),
            # take the first 4 video files
            rrb.Horizontal(
                contents=[
                    rrb.Tabs(
                        rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                        rrb.Spatial2DView(
                            origin=f"{video_log_path}".replace("video", "depth"),
                        ),
                        active_tab=active_tab,
                    )
                    for video_log_path in exo_video_log_paths[:4]
                ]
            ),
        ],
        row_shares=[3, 1],
    )
    additional_views = rrb.Vertical(
        contents=[
            rrb.Tabs(
                rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                rrb.Spatial2DView(origin=f"{video_log_path}".replace("video", "depth")),
                active_tab=active_tab,
            )
            for video_log_path in exo_video_log_paths[4:]
        ]
    )
    # do the last 4 videos
    contents = [main_view]
    if num_videos_to_log == 8:
        contents.append(additional_views)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=contents,
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


class RerunLogPaths(TypedDict):
    timeline_name: str
    parent_log_path: Path
    cam_log_path_list: list[Path]


@dataclass
class KeypointsContainer:
    """Container for include and exclude keypoints"""

    include_points: np.ndarray  # shape (n,2)
    exclude_points: np.ndarray  # shape (m,2)

    @classmethod
    def empty(cls) -> "KeypointsContainer":
        """Create an empty keypoints container"""
        return cls(include_points=np.zeros((0, 2), dtype=float), exclude_points=np.zeros((0, 2), dtype=float))

    def add_point(self, point: tuple[float, float], label: Literal["include", "exclude"]) -> None:
        """Add a point with the specified label"""
        point_array = np.array([point], dtype=float)
        if label == "include":
            self.include_points = (
                np.vstack([self.include_points, point_array]) if self.include_points.shape[0] > 0 else point_array
            )
        else:
            self.exclude_points = (
                np.vstack([self.exclude_points, point_array]) if self.exclude_points.shape[0] > 0 else point_array
            )

    def clear(self) -> None:
        """Clear all points"""
        self.include_points = np.zeros((0, 2), dtype=float)
        self.exclude_points = np.zeros((0, 2), dtype=float)


def log_pinhole_rec(
    rec: rr.RecordingStream,
    camera: PinholeParameters,
    cam_log_path: Path,
    image_plane_distance: float = 0.5,
    static: bool = False,
) -> None:
    """
    Logs the pinhole camera parameters and transformation data.

    Parameters:
    camera (PinholeParameters): The pinhole camera parameters including intrinsics and extrinsics.
    cam_log_path (Path): The path where the camera log will be saved.
    image_plane_distance (float, optional): The distance of the image plane from the camera. Defaults to 0.5.
    static (bool, optional): If True, the log data will be marked as static. Defaults to False.

    Returns:
    None
    """
    # camera intrinsics
    rec.log(
        f"{cam_log_path}/pinhole",
        rr.Pinhole(
            image_from_camera=camera.intrinsics.k_matrix,
            height=camera.intrinsics.height,
            width=camera.intrinsics.width,
            camera_xyz=getattr(
                rr.ViewCoordinates,
                camera.intrinsics.camera_conventions,
            ),
            image_plane_distance=image_plane_distance,
        ),
        static=static,
    )
    # camera extrinsics
    rec.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=camera.extrinsics.cam_t_world,
            mat3x3=camera.extrinsics.cam_R_world,
            from_parent=True,
        ),
        static=static,
    )


def log_video_rec(
    rec: rr.RecordingStream,
    video_path: Path,
    video_log_path: Path,
    timeline: str = "video_time",
) -> Int[np.ndarray, "num_frames"]:
    """
    Logs a video asset and its frame timestamps.

    Parameters:
    video_path (Path): The path to the video file.
    video_log_path (Path): The path where the video log will be saved.

    Returns:
    None
    """
    # Log video asset which is referred to by frame references.
    video_asset = rr.AssetVideo(path=video_path)
    rec.log(str(video_log_path), video_asset, static=True)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns: Int[np.ndarray, "num_frames"] = (  # noqa: UP037
        video_asset.read_frame_timestamps_ns()
    )
    rec.send_columns(
        f"{video_log_path}",
        # Note timeline values don't have to be the same as the video timestamps.
        indexes=[rr.TimeNanosColumn(timeline, frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanoseconds(frame_timestamps_ns),
    )
    return frame_timestamps_ns


# In this function, the `request` and `evt` parameters will be automatically injected by Gradio when this event listener is fired.
#
# `SelectionChange` is a subclass of `EventData`: https://www.gradio.app/docs/gradio/eventdata
# `gr.Request`: https://www.gradio.app/main/docs/gradio/request
def update_keypoints(
    active_recording_id: uuid.UUID,
    point_type: Literal["include", "exclude"],
    mv_keypoint_dict: dict[str, KeypointsContainer],
    log_paths: RerunLogPaths,
    request: gr.Request,
    evt: SelectionChange,
):
    if active_recording_id == "":
        return

    # We can only log a keypoint if the user selected only a single item.
    if len(evt.items) != 1:
        return
    item = evt.items[0]

    # If the selected item isn't an entity, or we don't have its position, then bail out.
    if item.kind != "entity" or item.position is None:
        return

    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(active_recording_id)
    stream: rr.BinaryStream = rec.binary_stream()
    current_keypoint: tuple[int, int] = item.position[0:2]

    for cam_name in mv_keypoint_dict:
        if cam_name in item.entity_path:
            # Update the keypoints for the specific camera
            mv_keypoint_dict[cam_name].add_point(current_keypoint, point_type)
            current_keypoint_container: KeypointsContainer = mv_keypoint_dict[cam_name]

    rec.set_time_nanos(log_paths["timeline_name"], nanos=0)
    # Log include points if any exist
    if current_keypoint_container.include_points.shape[0] > 0:
        rec.log(
            f"{item.entity_path}/include",
            rr.Points2D(current_keypoint_container.include_points, colors=(0, 255, 0), radii=5),
        )

    # Log exclude points if any exist
    if current_keypoint_container.exclude_points.shape[0] > 0:
        rec.log(
            f"{item.entity_path}/exclude",
            rr.Points2D(current_keypoint_container.exclude_points, colors=(255, 0, 0), radii=5),
        )

    # # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read(), mv_keypoint_dict


def get_recording(recording_id) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="multiview_sam_annotate", recording_id=recording_id)


def rescale_img(img_hw3: UInt8[np.ndarray, "h w 3"], max_dim: int) -> UInt8[np.ndarray, "... 3"]:
    # resize the image to have a max dim of max_dim
    height, width, _ = img_hw3.shape
    current_dim = max(height, width)

    # If current dimension is larger than max_dim, calculate scale factor
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Resize image maintaining aspect ratio
        resized_img = cv2.resize(img_hw3, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img

    # Return original image if no resize needed
    return img_hw3


def reset_keypoints(
    active_recording_id: uuid.UUID, mv_keypoint_dict: dict[str, KeypointsContainer], log_paths: RerunLogPaths
):
    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(active_recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    mv_keypoint_dict: dict[str, KeypointsContainer] = {
        cam_name: KeypointsContainer.empty() for cam_name in mv_keypoint_dict
    }

    rec.set_time_nanos(log_paths["timeline_name"], nanos=0)
    # Log include points if any exist
    for cam_log_path in log_paths["cam_log_path_list"]:
        pinhole_path: Path = cam_log_path / "pinhole"
        print(pinhole_path)
        rec.log(
            f"{pinhole_path}/video/include",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/video/exclude",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/video/bbox",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/video/bbox_center",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/segmentation",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/depth",
            rr.Clear(recursive=True),
        )

    rec.log(
        f"{log_paths['parent_log_path']}/triangulated",
        rr.Clear(recursive=True),
    )

    # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read(), mv_keypoint_dict, {}


def get_initial_mask(
    recording_id: uuid.UUID,
    inference_state: dict,
    mv_keypoints_dict: dict[str, KeypointsContainer],
    log_paths: RerunLogPaths,
    rgb_list: list[UInt8[np.ndarray, "h w 3"]],
    keypoint_centers_dict: dict[str, Float32[np.ndarray, "3"]],
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    rec.set_time_nanos(log_paths["timeline_name"], nanos=0)

    for (cam_name, keypoint_container), rgb in zip(mv_keypoints_dict.items(), rgb_list, strict=True):
        IMG_SAM_PREDICTOR.set_image(rgb)
        pinhole_log_path: Path = log_paths["parent_log_path"] / cam_name / "pinhole"
        points: Float32[np.ndarray, "num_points 2"] = np.vstack(
            [keypoint_container.include_points, keypoint_container.exclude_points]
        ).astype(np.float32)
        if points.shape[0] == 0:
            IMG_SAM_PREDICTOR.reset_predictor()
            rec.log(
                "logs",
                rr.TextLog("No points selected, skipping segmentation.", level="info"),
            )
        else:
            # Create labels array: 1 for include points, 0 for exclude points
            labels: Int[np.ndarray, "num_points"] = np.ones(len(keypoint_container.include_points), dtype=np.int32)
            if len(keypoint_container.exclude_points) > 0:
                labels = np.concatenate([labels, np.zeros(len(keypoint_container.exclude_points), dtype=np.int32)])

            with torch.inference_mode():
                masks, scores, _ = IMG_SAM_PREDICTOR.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False,
                )
                masks: Bool[np.ndarray, "1 h w"] = masks > 0.0

            rec.log(
                f"{pinhole_log_path}/segmentation",
                rr.SegmentationImage(masks[0].astype(np.uint8)),
            )
            # Convert the mask to a bounding box
            if masks[0].any():
                y_min, y_max = np.where(masks[0].any(axis=1))[0][[0, -1]]
                x_min, x_max = np.where(masks[0].any(axis=0))[0][[0, -1]]
                bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                rec.log(
                    f"{pinhole_log_path}/video/bbox",
                    rr.Boxes2D(array=bbox, array_format=rr.Box2DFormat.XYXY, colors=(0, 0, 255)),
                )

                # Calculate the center of the bounding box
                center_xyc: Float32[np.ndarray, "3"] = np.array(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, 1], dtype=np.float32
                )
                rec.log(
                    f"{pinhole_log_path}/video/bbox_center",
                    rr.Points2D(positions=(center_xyc[0], center_xyc[1]), colors=(0, 0, 255), radii=5),
                )
                keypoint_centers_dict[cam_name] = center_xyc
            IMG_SAM_PREDICTOR.reset_predictor()

        yield stream.read(), keypoint_centers_dict


def triangulate_centers(
    recording_id: uuid.UUID,
    center_xyc_dict: dict[str, Float32[np.ndarray, "3"]],
    exo_cam_list: list[PinholeParameters],
    log_paths: RerunLogPaths,
    rgb_list: list[UInt8[np.ndarray, "h w 3"]],
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    masks_list: list[UInt8[np.ndarray, "h w"]] = []

    rec.set_time_nanos(log_paths["timeline_name"], nanos=0)
    if len(center_xyc_dict) >= 2:
        centers_xyc: Float32[np.ndarray, "num_views 3"] = np.stack(
            [center_xyc for center_xyc in center_xyc_dict.values() if center_xyc is not None], axis=0
        ).astype(np.float32)
        centers_xyc = rearrange(centers_xyc, "num_views xyc -> num_views 1 xyc")
        proj_matrices: list[Float32[np.ndarray, "3 4"]] = [exo_cam.projection_matrix for exo_cam in exo_cam_list]
        proj_matrices: Float32[np.ndarray, "num_views 3 4"] = np.stack(proj_matrices, axis=0).astype(np.float32)

        proj_matrices_filtered: list[Float32[np.ndarray, "3 4"]] = [
            exo_cam.projection_matrix for exo_cam in exo_cam_list if exo_cam.name in center_xyc_dict
        ]
        proj_matrices_filtered: Float32[np.ndarray, "num_views 3 4"] = np.stack(proj_matrices_filtered, axis=0).astype(
            np.float32
        )
        xyzc: Float[np.ndarray, "n_points 4"] = batch_triangulate(
            keypoints_2d=centers_xyc, projection_matrices=proj_matrices_filtered
        )
        rec.log(
            f"{log_paths['parent_log_path']}/triangulated", rr.Points3D(xyzc[:, 0:3], colors=(0, 0, 255), radii=0.1)
        )

        projected_xyc = projectN3(
            xyzc,
            proj_matrices,
        )

        for rgb, cam_log_path, xyc in zip(rgb_list, log_paths["cam_log_path_list"], projected_xyc, strict=True):
            pinhole_log_path: Path = cam_log_path / "pinhole"
            xy = xyc[:, 0:2]
            rec.log(
                f"{pinhole_log_path}/video/bbox_center",
                rr.Points2D(positions=xy, colors=(0, 0, 255), radii=5),
            )
            IMG_SAM_PREDICTOR.set_image(rgb)
            labels: Int[np.ndarray, "num_points"] = np.ones(len(xyc), dtype=np.int32)
            with torch.inference_mode():
                masks, scores, _ = IMG_SAM_PREDICTOR.predict(
                    point_coords=xy,
                    point_labels=labels,
                    multimask_output=False,
                )
                masks: Bool[np.ndarray, "1 h w"] = masks > 0.0

            mask = masks[0].astype(np.uint8)
            masks_list.append(mask)
            rec.log(
                f"{pinhole_log_path}/segmentation",
                rr.SegmentationImage(mask),
            )
            if mask.any():
                y_min, y_max = np.where(masks[0].any(axis=1))[0][[0, -1]]
                x_min, x_max = np.where(masks[0].any(axis=0))[0][[0, -1]]
                bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                rec.log(
                    f"{pinhole_log_path}/video/bbox",
                    rr.Boxes2D(array=bbox, array_format=rr.Box2DFormat.XYXY, colors=(0, 0, 255)),
                )

                # Calculate the center of the bounding box
                center_xyc: Float32[np.ndarray, "3"] = np.array(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, 1], dtype=np.float32
                )
                rec.log(
                    f"{pinhole_log_path}/video/bbox_center",
                    rr.Points2D(positions=(center_xyc[0], center_xyc[1]), colors=(0, 0, 255), radii=5),
                )
            IMG_SAM_PREDICTOR.reset_predictor()

    else:
        rec.log(
            "logs",
            rr.TextLog("No points selected, skipping segmentation.", level="info"),
        )
        gr.Info("Not enough points to triangulate.")
    yield stream.read(), masks_list


def log_dataset(dataset_name: Literal["hocap", "assembly101"]):
    recording_id: uuid.UUID = uuid.uuid4()
    rec: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    match dataset_name:
        case "hocap":
            sequence: HOCapSequence = HOCapSequence(
                data_path=Path("data/hocap/sample"),
                sequence_name="20231024_180733",
                subject_id="8",
                load_labels=False,
            )
        case "assembly101":
            # raise NotImplementedError("Assembly101 is not implemented yet.")
            sequence: Assembely101Sequence = Assembely101Sequence(
                data_path=Path("data/assembly101-sample"),
                sequence_name="nusar-2021_action_both_9015-b05b_9015_user_id_2021-02-02_161800",
                subject_id=None,
                load_labels=False,
            )
        case _:
            assert_never(dataset_name)

    parent_log_path: Path = Path("world")
    timeline_name: str = "frame_idx"

    images_to_log: int = 8

    exo_video_readers: MultiVideoReader = sequence.exo_video_readers
    # exo_video_files: list[Path] = exo_video_readers.video_paths[0:images_to_log]
    exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in sequence.exo_cam_list][
        0:images_to_log
    ]
    exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths][
        0:images_to_log
    ]

    initial_blueprint = create_blueprint(exo_video_log_paths, num_videos_to_log=8)
    rec.send_blueprint(initial_blueprint)
    rec.log("/", sequence.world_coordinate_system, static=True)

    bgr_list: list[UInt8[np.ndarray, "h w 3"]] = exo_video_readers[0][0:images_to_log]
    rgb_list: list[UInt8[np.ndarray, "h w 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]
    # check if depth images exist
    if not sequence.depth_paths:
        depth_paths = None
    else:
        depth_paths: dict[ExoCameraIDs, Path] = sequence.depth_paths[0]
    exo_cam_list: list[PinholeParameters] = sequence.exo_cam_list[0:images_to_log]

    cam_log_path_list: list[Path] = []
    fuser = Open3DFuser(fusion_resolution=0.01, max_fusion_depth=1.25)
    # log stationary exo cameras and video assets
    for exo_cam in exo_cam_list:
        cam_log_path: Path = parent_log_path / exo_cam.name
        cam_log_path_list.append(cam_log_path)
        image_plane_distance: float = 0.1 if dataset_name == "hocap" else 100.0
        log_pinhole_rec(
            rec=rec,
            camera=exo_cam,
            cam_log_path=cam_log_path,
            image_plane_distance=image_plane_distance,
            static=True,
        )

    for rgb, cam_log_path, exo_cam in zip(rgb_list, cam_log_path_list, exo_cam_list, strict=True):
        pinhole_log_path: Path = cam_log_path / "pinhole"
        rec.log(f"{pinhole_log_path}/video", rr.Image(rgb, color_model=rr.ColorModel.RGB), static=True)
        # rec.log(f"{pinhole_log_path}/depth", rr.DepthImage(depth_image, meter=1000))
        if depth_paths is not None:
            depth_path: Path = depth_paths[cam_log_path.name]
            depth_image: UInt16[np.ndarray, "480 640"] = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            fuser.fuse_frames(
                depth_image,
                exo_cam.intrinsics.k_matrix,
                exo_cam.extrinsics.cam_T_world,
                rgb,
            )

    if depth_paths is not None:
        mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
        mesh.compute_vertex_normals()

        rec.log(
            f"{parent_log_path}/mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.triangles,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=mesh.vertex_colors,
            ),
            static=True,
        )

        pcd: o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(
            number_of_points=20_000,
        )

    log_paths = RerunLogPaths(
        timeline_name=timeline_name,
        parent_log_path=parent_log_path,
        cam_log_path_list=cam_log_path_list,
    )

    mv_keypoint_dict: dict[str, KeypointsContainer] = {
        cam_log_path.name: KeypointsContainer.empty() for cam_log_path in cam_log_path_list
    }

    yield stream.read(), recording_id, log_paths, mv_keypoint_dict, rgb_list, exo_cam_list, pcd


def handle_export(
    exo_cam_list: list[PinholeParameters],
    rgb_list: list[UInt8[np.ndarray, "h w 3"]],
    masks_list: list[UInt8[np.ndarray, "h w"]],
    pointcloud: o3d.geometry.PointCloud,
):
    bgr_list: list[UInt8[np.ndarray, "h w 3"]] = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in rgb_list]
    ns_save_dir: Path = Path("data/nerfstudio-export")
    masks_list: list[UInt8[np.ndarray, "h w"]] | None = masks_list if len(masks_list) > 0 else None
    save_to_nerfstudio(
        ns_save_path=ns_save_dir,
        pinhole_param_list=exo_cam_list,
        bgr_list=bgr_list,
        pointcloud=pointcloud,
        masks_list=masks_list,
    )

    # Define the path for the output zip file
    zip_output_path = Path("data/nerfstudio-output")
    zip_file_path: str = shutil.make_archive(str(zip_output_path), "zip", str(ns_save_dir))

    # Return the path to the zip file and switch tabs
    return gr.Tabs(selected=1), zip_file_path


with gr.Blocks() as mv_sam_block:
    mv_keypoint_dict: dict[str, KeypointsContainer] = gr.State({})
    inference_state = gr.State({})
    rgb_list = gr.State()
    masks_list: list[UInt8[np.ndarray, "h w"]] = gr.State([])
    exo_cam_list: list[PinholeParameters] = gr.State([])
    pointcloud: o3d.geometry.PointCloud = gr.State()
    centers_xyc_dict: dict[str, Float32[np.ndarray, "3"]] = gr.State({})
    with gr.Row():
        with gr.Tabs() as main_tabs:
            with gr.TabItem("Controls", id=0):
                with gr.Column(scale=1):
                    dataset_dropdown = gr.Dropdown(
                        label="Dataset",
                        choices=["hocap", "assembly101"],
                        value="hocap",
                    )
                    load_dataset_btn = gr.Button("Load Dataset")

                    point_type = gr.Radio(
                        label="point type",
                        choices=["include", "exclude"],
                        value="include",
                        scale=1,
                    )
                    clear_points_btn = gr.Button("Clear Points", scale=1)
                    get_initial_mask_btn = gr.Button("Get Initial Mask", scale=1)
                    triangulate_btn = gr.Button("Triangulate Center", scale=1)
                    export_btn = gr.Button("Export", scale=1)
            with gr.TabItem("Output", id=1):
                gr.Markdown("here you can see the output of the selected video")
                output_zip = gr.File(label="Exported Zip File", file_count="single", type="filepath")
        with gr.Column(scale=4):
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=700,
            )

    # We make a new recording id, and store it in a Gradio's session state.
    recording_id = gr.State()
    log_paths = gr.State({})

    load_dataset_btn.click(
        fn=log_dataset,
        inputs=[dataset_dropdown],
        outputs=[viewer, recording_id, log_paths, mv_keypoint_dict, rgb_list, exo_cam_list, pointcloud],
    )

    viewer.selection_change(
        update_keypoints,
        inputs=[
            recording_id,
            point_type,
            mv_keypoint_dict,
            log_paths,
        ],
        outputs=[viewer, mv_keypoint_dict],
    )

    clear_points_btn.click(
        fn=reset_keypoints,
        inputs=[recording_id, mv_keypoint_dict, log_paths],
        outputs=[viewer, mv_keypoint_dict, centers_xyc_dict],
    )

    get_initial_mask_btn.click(
        fn=get_initial_mask,
        inputs=[recording_id, inference_state, mv_keypoint_dict, log_paths, rgb_list, centers_xyc_dict],
        outputs=[viewer, centers_xyc_dict],
    )

    triangulate_btn.click(
        fn=triangulate_centers,
        inputs=[recording_id, centers_xyc_dict, exo_cam_list, log_paths, rgb_list],
        outputs=[viewer, masks_list],
    )
    # TODO export masks + ply + camera poses for use with brush
    export_btn.click(
        fn=handle_export,
        inputs=[exo_cam_list, rgb_list, masks_list, pointcloud],
        outputs=[main_tabs, output_zip],
    )
