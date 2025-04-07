import copy
import tempfile
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import cv2
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from gradio_rerun import Rerun
from gradio_rerun.events import (
    SelectionChange,
)
from jaxtyping import Bool, Float, Float32, Int, UInt8
from monopriors.depth_utils import clip_disparity, depth_edges_mask, depth_to_points
from monopriors.relative_depth_models.depth_anything_v2 import (
    DepthAnythingV2Predictor,
    RelativeDepthPrediction,
)
from sam2.sam2_video_predictor import SAM2VideoPredictor
from simplecv.apis.view_exoego_data import create_blueprint as create_exoego_blueprint
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exoego.hocap import HOCapSequence
from simplecv.video_io import MultiVideoReader, VideoReader

from sam2_depthanything.op import create_blueprint

if gr.NO_RELOAD:
    VIDEO_SAM_PREDICTOR: SAM2VideoPredictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    DEPTH_PREDICTOR = DepthAnythingV2Predictor(device="cpu", encoder="vits")
    DEPTH_PREDICTOR.set_model_device("cuda")


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


def log_relative_pred_rec(
    rec: rr.RecordingStream,
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
    cam_T_world_44: Float[np.ndarray, "4 4"] = np.eye(4)

    rec.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=cam_T_world_44[:3, 3],
            mat3x3=cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rec.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=relative_pred.K_33,
            width=rgb_hw3.shape[1],
            height=rgb_hw3.shape[0],
            image_plane_distance=1.5,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rec.log(f"{pinhole_path}/image", rr.Image(rgb_hw3).compress(jpeg_quality=jpeg_quality))

    depth_hw: Float32[np.ndarray, "h w"] = relative_pred.depth
    disparity = relative_pred.disparity
    # removes outliers from disparity (sometimes we can get weirdly large values)
    clipped_disparity: UInt8[np.ndarray, "h w"] = clip_disparity(disparity)
    if remove_flying_pixels:
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=depth_edge_threshold)
        rec.log(
            f"{pinhole_path}/edge_mask",
            rr.SegmentationImage(edges_mask.astype(np.uint8)),
        )
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw * ~edges_mask
        clipped_disparity: Float32[np.ndarray, "h w"] = clipped_disparity * ~edges_mask

    if seg_mask_hw is not None:
        rec.log(
            f"{pinhole_path}/segmentation",
            rr.SegmentationImage(seg_mask_hw),
        )
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw  # * seg_mask_hw
        clipped_disparity: Float32[np.ndarray, "h w"] = clipped_disparity  # * seg_mask_hw

    rec.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw))

    # log to cam_log_path to avoid backprojecting disparity
    rec.log(f"{cam_log_path}/disparity", rr.DepthImage(clipped_disparity))

    depth_1hw: Float32[np.ndarray, "1 h w"] = rearrange(depth_hw, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, relative_pred.K_33)

    colors = rgb_hw3.reshape(-1, 3)

    # If we have a segmentation mask, make those pixels blue
    if seg_mask_hw is not None:
        # Reshape the mask to match colors shape
        flat_mask = seg_mask_hw.reshape(-1)

        # Set pixels where mask == 1 to blue (BGR format)
        # Blue: [255, 0, 0] in BGR or [0, 0, 255] in RGB
        colors[flat_mask == 1, :] = [0, 0, 255]  # RGB format: Blue

    rec.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=colors,
        ),
    )


# In this function, the `request` and `evt` parameters will be automatically injected by Gradio when this event listener is fired.
#
# `SelectionChange` is a subclass of `EventData`: https://www.gradio.app/docs/gradio/eventdata
# `gr.Request`: https://www.gradio.app/main/docs/gradio/request
def update_keypoints(
    active_recording_id: uuid.UUID,
    point_type: Literal["include", "exclude"],
    keypoints_container: KeypointsContainer,
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
    keypoints_container.add_point(current_keypoint, point_type)

    rec.set_time_nanos(log_paths["timeline_name"], nanos=0)
    # Log include points if any exist
    if keypoints_container.include_points.shape[0] > 0:
        rec.log(
            f"{item.entity_path}/include", rr.Points2D(keypoints_container.include_points, colors=(0, 255, 0), radii=5)
        )

    # Log exclude points if any exist
    if keypoints_container.exclude_points.shape[0] > 0:
        rec.log(
            f"{item.entity_path}/exclude",
            rr.Points2D(keypoints_container.exclude_points, colors=(255, 0, 0), radii=5),
        )

    # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read(), keypoints_container


def get_recording(recording_id) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="rerun_vggt_sam", recording_id=recording_id)


# Allow using keyword args in gradio to avoid mixing up the order of inputs
@dataclass
class InputComponents:
    video_file: gr.Video

    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class InputValues:
    video_file: str


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


def preprocess_video(
    *input_params,
    progress=gr.Progress(track_tqdm=True),  # noqa B008
):
    input_values = InputValues(*input_params)
    # create a new recording id, and store it in a Gradio's session state.
    recording_id: uuid.UUID = uuid.uuid4()
    rec: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    log_paths = RerunLogPaths(
        timeline_name="frame_idx",
        parent_log_path=Path("world"),
        camera_log_path=Path("world") / "camera",
        pinhole_path=Path("world") / "camera" / "pinhole",
    )

    video_path: Path = Path(input_values.video_file)

    initial_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{log_paths['pinhole_path']}"),
        ),
        collapse_panels=True,
    )

    rec.send_blueprint(initial_blueprint)

    video_reader: VideoReader = VideoReader(video_path)
    tmp_frames_dir: str = tempfile.mkdtemp()

    target_fps: int = 10
    frame_interval: int = int(video_reader.fps // target_fps)
    max_frames: int = 100
    total_saved_frames: int = 0
    max_size: int = 640

    progress(0, desc="Reading video frames")
    for idx, bgr in enumerate(video_reader):
        if idx % frame_interval == 0:
            if total_saved_frames >= max_frames:
                break
            bgr: np.ndarray = rescale_img(bgr, max_size)
            # 3. Save frames to temporary directory
            cv2.imwrite(f"{tmp_frames_dir}/{idx:05d}.jpg", bgr)
            total_saved_frames += 1

    first_frame_path: Path = Path(tmp_frames_dir) / "00000.jpg"
    first_bgr: np.ndarray = cv2.imread(str(first_frame_path))

    progress(0.5, desc="Initializing SAM")
    with torch.inference_mode():
        inference_state = VIDEO_SAM_PREDICTOR.init_state(video_path=tmp_frames_dir)
        VIDEO_SAM_PREDICTOR.reset_state(inference_state)
    print(type(inference_state))

    rec.set_time_sequence(log_paths["timeline_name"], sequence=0)
    rec.log(
        f"{log_paths['pinhole_path']}/image",
        rr.Image(first_bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=90),
    )

    # Ensure we consume everything from the recording.
    stream.flush()

    yield gr.Accordion(open=False), stream.read(), inference_state, Path(tmp_frames_dir), recording_id, log_paths


def reset_keypoints(active_recording_id: uuid.UUID, keypoints_container: KeypointsContainer, log_paths: RerunLogPaths):
    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(active_recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    keypoints_container.clear()

    rec.set_time_sequence(log_paths["timeline_name"], sequence=0)
    # Log include points if any exist
    # paths_to_clear = ["include", "exclude", "segmentation", "depth", "image"]
    # for path in paths_to_clear:
    #     rec.log(
    #         f"{log_paths['pinhole_path']}/{path}",
    #         rr.Clear(recursive=True),
    #     )
    rec.log(
        f"{log_paths['pinhole_path']}/image/include",
        rr.Clear(recursive=True),
    )
    rec.log(
        f"{log_paths['pinhole_path']}/image/exclude",
        rr.Clear(recursive=True),
    )
    rec.log(
        f"{log_paths['pinhole_path']}/segmentation",
        rr.Clear(recursive=True),
    )
    rec.log(
        f"{log_paths['pinhole_path']}/depth",
        rr.Clear(recursive=True),
    )

    # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read(), keypoints_container


def get_initial_mask(
    recording_id: uuid.UUID,
    inference_state: dict,
    keypoint_container: KeypointsContainer,
    log_paths: RerunLogPaths,
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    rec.set_time_sequence(log_paths["timeline_name"], 0)

    points = np.vstack([keypoint_container.include_points, keypoint_container.exclude_points]).astype(np.float32)
    if len(points) == 0:
        raise gr.Error("No points selected. Please add include or exclude points.")

    # Create labels array: 1 for include points, 0 for exclude points
    labels = np.ones(len(keypoint_container.include_points), dtype=np.int32)
    if len(keypoint_container.exclude_points) > 0:
        labels = np.concatenate([labels, np.zeros(len(keypoint_container.exclude_points), dtype=np.int32)])

    print(f"Points shape: {points.shape}")
    print(f"Labels shape: {labels.shape}")
    print(labels)
    print(
        f"Include points: {keypoint_container.include_points.shape}, Exclude points: {keypoint_container.exclude_points.shape}"
    )

    with torch.inference_mode():
        frame_idx: int
        object_ids: list
        masks: Float32[torch.Tensor, "b 3 h w"]

        frame_idx, object_ids, masks = VIDEO_SAM_PREDICTOR.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=0,
            points=points,
            labels=labels,
        )

        masks: Bool[np.ndarray, "1 h w"] = (masks[0] > 0.0).numpy(force=True)

    rec.log(
        f"{log_paths['pinhole_path']}/segmentation",
        rr.SegmentationImage(masks[0].astype(np.uint8)),
    )
    yield stream.read()


def propagate_mask(
    recording_id: uuid.UUID,
    inference_state: dict,
    keypoint_container: KeypointsContainer,
    frames_dir: Path,
    log_paths: RerunLogPaths,
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    blueprint = create_blueprint(parent_log_path=log_paths["parent_log_path"])
    rec.send_blueprint(blueprint)

    rec.log(f"{log_paths['parent_log_path']}", rr.ViewCoordinates.RDF)

    points = np.vstack([keypoint_container.include_points, keypoint_container.exclude_points]).astype(np.float32)
    if len(points) == 0:
        raise gr.Error("No points selected. Please add include or exclude points.")

    # Create labels array: 1 for include points, 0 for exclude points
    labels = np.ones(len(keypoint_container.include_points), dtype=np.int32)
    if len(keypoint_container.exclude_points) > 0:
        labels = np.concatenate([labels, np.zeros(len(keypoint_container.exclude_points), dtype=np.int32)])

    frames_paths: list[Path] = sorted(frames_dir.glob("*.jpg"))

    # remove the keypoints as they're in the way during propagation
    rec.log(
        f"{log_paths['pinhole_path']}/include",
        rr.Clear(recursive=True),
    )
    rec.log(
        f"{log_paths['pinhole_path']}/exclude",
        rr.Clear(recursive=True),
    )

    with torch.inference_mode():
        frame_idx: int
        object_ids: list
        masks: Float32[torch.Tensor, "b 3 h w"]

        frame_idx, object_ids, masks = VIDEO_SAM_PREDICTOR.add_new_points_or_box(
            inference_state, frame_idx=0, obj_id=0, points=points, labels=labels
        )

        # propagate the prompts to get masklets throughout the video
        for frames_path, (frame_idx, object_ids, masks) in zip(
            frames_paths, VIDEO_SAM_PREDICTOR.propagate_in_video(inference_state), strict=True
        ):
            rec.set_time_sequence(log_paths["timeline_name"], frame_idx)
            masks: Bool[np.ndarray, "1 h w"] = (masks[0] > 0.0).numpy(force=True)
            bgr = cv2.imread(str(frames_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_pred: RelativeDepthPrediction = DEPTH_PREDICTOR.__call__(rgb=rgb, K_33=None)

            log_relative_pred_rec(
                rec=rec,
                parent_log_path=log_paths["parent_log_path"],
                relative_pred=depth_pred,
                rgb_hw3=rgb,
                seg_mask_hw=masks[0].astype(np.uint8),
                remove_flying_pixels=True,
                jpeg_quality=90,
                depth_edge_threshold=0.1,
            )

            yield stream.read()


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
            raise gr.Error("Assembly101 dataset is not supported yet.")
        case _:
            assert_never(dataset_name)

    parent_log_path: Path = Path("world")
    timeline_name: str = "frame_idx"

    exo_video_readers: MultiVideoReader = sequence.exo_video_readers
    exo_video_files: list[Path] = exo_video_readers.video_paths
    exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in sequence.exo_cam_list]
    exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

    initial_blueprint = create_exoego_blueprint(exo_video_log_paths, num_videos_to_log=8)
    rec.send_blueprint(initial_blueprint)

    cam_log_path_list: list[Path] = []
    # log stationary exo cameras and video assets
    for exo_cam in sequence.exo_cam_list:
        cam_log_path: Path = parent_log_path / exo_cam.name
        cam_log_path_list.append(cam_log_path)
        image_plane_distance: float = 0.1
        log_pinhole_rec(
            rec=rec,
            camera=exo_cam,
            cam_log_path=cam_log_path,
            image_plane_distance=image_plane_distance,
            static=True,
        )

    log_paths = RerunLogPaths(
        timeline_name=timeline_name,
        parent_log_path=parent_log_path,
        cam_log_path_list=cam_log_path_list,
    )

    all_timestamps: list[Int[np.ndarray, "num_frames"]] = []  # noqa: UP037
    for video_file, video_log_path in zip(exo_video_files, exo_video_log_paths, strict=True):
        assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
        # Log video asset which is referred to by frame references.
        frame_timestamps_ns: Int[np.ndarray, "num_frames"] = log_video_rec(  # noqa: UP037
            rec=rec,
            video_path=video_file,
            video_log_path=video_log_path,
            timeline=log_paths["timeline_name"],
        )
        all_timestamps.append(frame_timestamps_ns)

    yield stream.read(), recording_id, log_paths


with gr.Blocks() as mv_sam_block:
    keypoints = gr.State(KeypointsContainer.empty())
    inference_state = gr.State({})
    frames_dir = gr.State(Path())
    with gr.Row():
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
            propagate_mask_btn = gr.Button("Propagate Mask", scale=1)
            stop_propagation_btn = gr.Button("Stop Propagation", scale=1)

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

    # input_components = InputComponents(
    #     video_file=video_in,
    # )

    # triggered on video upload
    # video_in.upload(
    #     fn=preprocess_video,
    #     inputs=input_components.to_list(),
    #     outputs=[video_in_drawer, viewer, inference_state, frames_dir, recording_id, log_paths],
    # )

    load_dataset_btn.click(
        fn=log_dataset,
        inputs=[dataset_dropdown],
        outputs=[viewer, recording_id, log_paths],
    )

    viewer.selection_change(
        update_keypoints,
        inputs=[
            recording_id,
            point_type,
            keypoints,
            log_paths,
        ],
        outputs=[viewer, keypoints],
    )

    # clear_points_btn.click(
    #     fn=reset_keypoints,
    #     inputs=[recording_id, keypoints, log_paths],
    #     outputs=[viewer, keypoints],
    # )

    # get_initial_mask_btn.click(
    #     fn=get_initial_mask,
    #     inputs=[recording_id, inference_state, keypoints, log_paths],
    #     outputs=[viewer],
    # )

    # propagate_event = propagate_mask_btn.click(
    #     fn=propagate_mask,
    #     inputs=[recording_id, inference_state, keypoints, frames_dir, log_paths],
    #     outputs=[viewer],
    # )

    # stop_propagation_btn.click(
    #     fn=lambda: None,
    #     inputs=[],
    #     outputs=[],
    #     cancels=[propagate_event],
    # )
