try:
    import spaces  # type: ignore

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False


import math
import tempfile
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import cv2
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from gradio_rerun import Rerun
from gradio_rerun.events import (
    SelectionChange,
    TimelineChange,
    TimeUpdate,
)
from jaxtyping import Bool, Float32, Int, UInt8
from monopriors.relative_depth_models.depth_anything_v2 import (
    DepthAnythingV2Predictor,
)
from numpy import ndarray
from sam2.sam2_video_predictor import SAM2VideoPredictor
from simplecv.video_io import VideoReader

if gr.NO_RELOAD:
    VIDEO_SAM_PREDICTOR: SAM2VideoPredictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")


def log_video_rec(
    rec: rr.RecordingStream, video_path: Path, video_log_path: Path, timeline: str
) -> Int[ndarray, "num_frames"]:
    # Log video asset which is referred to by frame references.
    video_asset = rr.AssetVideo(path=video_path)
    rec.log(f"{video_log_path}", video_asset, static=True)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns: Int[ndarray, "num_frames"] = (  # noqa: UP037
        video_asset.read_frame_timestamps_ns()
    )
    rec.send_columns(
        f"{video_log_path}",
        # Note timeline values don't have to be the same as the video timestamps.
        indexes=[rr.TimeNanosColumn(timeline, frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanoseconds(frame_timestamps_ns),
    )
    return frame_timestamps_ns


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


# In this function, the `request` and `evt` parameters will be automatically injected by Gradio when this event listener is fired.
#
# `SelectionChange` is a subclass of `EventData`: https://www.gradio.app/docs/gradio/eventdata
# `gr.Request`: https://www.gradio.app/main/docs/gradio/request
def update_keypoints(
    active_recording_id: uuid.UUID,
    point_type: Literal["include", "exclude"],
    keypoints_container: KeypointsContainer,
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

    rec.set_time_sequence("video_time", sequence=0)
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


# weird nesting is to avoid issues with beartype checking
def preprocess_video(
    *input_params,
):  # <-- intentionally unannotated, because gradio hates @beartype
    yield from _preprocess_video(
        *input_params,
        progress=gr.Progress(track_tqdm=True),  # noqa B008
    )  # <-- magic happens


def _preprocess_video(
    *input_params,
    progress=gr.Progress(track_tqdm=True),  # noqa B008
):
    input_values = InputValues(*input_params)
    # create a new recording id, and store it in a Gradio's session state.
    recording_id: uuid.UUID = uuid.uuid4()
    rec: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    parent_log_path = Path("world")
    video_log_path = parent_log_path / "video"
    video_path: Path = Path(input_values.video_file)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{video_log_path}"),
        ),
        collapse_panels=True,
    )

    rec.send_blueprint(blueprint)

    video_reader: VideoReader = VideoReader(video_path)
    tmp_dir: str = tempfile.mkdtemp()

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
            cv2.imwrite(f"{tmp_dir}/{idx:05d}.jpg", bgr)
            total_saved_frames += 1

    first_frame_path: Path = Path(tmp_dir) / "00000.jpg"
    first_bgr: np.ndarray = cv2.imread(str(first_frame_path))

    progress(0.5, desc="Initializing SAM")
    with torch.inference_mode():
        inference_state = VIDEO_SAM_PREDICTOR.init_state(video_path=tmp_dir)
        VIDEO_SAM_PREDICTOR.reset_state(inference_state)
    print(type(inference_state))

    rec.set_time_sequence("video_time", sequence=0)
    rec.log(
        f"{video_log_path}",
        rr.Image(first_bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=90),
    )

    # Ensure we consume everything from the recording.
    stream.flush()

    yield gr.Accordion(open=False), stream.read(), inference_state, Path(tmp_dir), recording_id


def reset_keypoints(active_recording_id: uuid.UUID, keypoints_container: KeypointsContainer):
    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(active_recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    keypoints_container.clear()

    rec.set_time_sequence("video_time", sequence=0)
    # Log include points if any exist
    rec.log(
        "world/video/include",
        rr.Clear(recursive=True),
    )
    rec.log(
        "world/video/exclude",
        rr.Clear(recursive=True),
    )
    rec.log(
        "world/video/mask",
        rr.Clear(recursive=True),
    )

    # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read(), keypoints_container


def get_initial_mask(recording_id, inference_state, keypoint_container):
    yield from _get_initial_mask(recording_id, inference_state, keypoint_container)  # <-- magic happens


def _get_initial_mask(
    recording_id: uuid.UUID,
    inference_state: dict,
    keypoint_container: KeypointsContainer,
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    parent_log_path = Path("world")
    rec.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)
    rec.set_time_sequence("video_time", 0)

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
        f"{parent_log_path}/video/mask",
        rr.SegmentationImage(masks[0].astype(np.uint8)),
    )
    yield stream.read()


def propagate_mask(recording_id, inference_state, keypoint_container, frames_dir):
    yield from _propagate_mask(recording_id, inference_state, keypoint_container, frames_dir)  # <-- magic happens


def _propagate_mask(
    recording_id: uuid.UUID, inference_state: dict, keypoint_container: KeypointsContainer, frames_dir: Path
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    points = np.vstack([keypoint_container.include_points, keypoint_container.exclude_points]).astype(np.float32)
    if len(points) == 0:
        raise gr.Error("No points selected. Please add include or exclude points.")

    # Create labels array: 1 for include points, 0 for exclude points
    labels = np.ones(len(keypoint_container.include_points), dtype=np.int32)
    if len(keypoint_container.exclude_points) > 0:
        labels = np.concatenate([labels, np.zeros(len(keypoint_container.exclude_points), dtype=np.int32)])

    parent_log_path = Path("world")
    frames_paths: list[Path] = sorted(frames_dir.glob("*.jpg"))

    # remove the keypoints as they're in the way during propagation
    rec.log(
        "world/video/include",
        rr.Clear(recursive=True),
    )
    rec.log(
        "world/video/exclude",
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
            rec.set_time_sequence("video_time", frame_idx)
            masks: Bool[np.ndarray, "1 h w"] = (masks[0] > 0.0).numpy(force=True)
            bgr = cv2.imread(str(frames_path))

            rec.log(
                f"{parent_log_path}/video",
                rr.Image(bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=90),
            )
            rec.log(
                f"{parent_log_path}/video/mask",
                rr.SegmentationImage(masks[0].astype(np.uint8)),
            )

            yield stream.read()


with gr.Blocks() as vggt_block:
    with gr.Tab("Monocular"):
        keypoints = gr.State(KeypointsContainer.empty())
        inference_state = gr.State({})
        frames_dir = gr.State(Path())
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Your video IN", open=True) as video_in_drawer:
                    video_in = gr.Video(label="Video IN", format=None)

                point_type = gr.Radio(
                    label="point type",
                    choices=["include", "exclude"],
                    value="include",
                    scale=1,
                )
                clear_points_btn = gr.Button("Clear Points", scale=1)
                get_initial_mask_btn = gr.Button("Get Initial Mask", scale=1)
                propagate_mask_btn = gr.Button("Propagate Mask", scale=1)

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

        input_components = InputComponents(
            video_file=video_in,
        )

        # triggered on video upload
        video_in.upload(
            fn=preprocess_video,
            inputs=input_components.to_list(),
            outputs=[video_in_drawer, viewer, inference_state, frames_dir, recording_id],
        )

        viewer.selection_change(
            update_keypoints,
            inputs=[
                recording_id,
                point_type,
                keypoints,
            ],
            outputs=[viewer, keypoints],
        )

        clear_points_btn.click(
            fn=reset_keypoints,
            inputs=[recording_id, keypoints],
            outputs=[viewer, keypoints],
        )

        get_initial_mask_btn.click(
            fn=get_initial_mask,
            inputs=[recording_id, inference_state, keypoints],
            outputs=[viewer],
        )

        propagate_mask_btn.click(
            fn=propagate_mask,
            inputs=[recording_id, inference_state, keypoints, frames_dir],
            outputs=[viewer],
        )
