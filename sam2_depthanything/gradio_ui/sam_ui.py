try:
    import spaces  # type: ignore

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

import gradio as gr
from gradio_rerun import Rerun

import cv2
import rerun as rr
from PIL import Image
from typing import Literal, no_type_check
import numpy as np

from sam2_depthanything.op import log_relative_pred, create_blueprint

from jaxtyping import UInt8, Float32, Bool
import tempfile
import mmcv
from mmcv.video import VideoReader
from pathlib import Path
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
from monopriors.relative_depth_models.depth_anything_v2 import (
    DepthAnythingV2Predictor,
    RelativeDepthPrediction,
)


if gr.NO_RELOAD:
    DEPTH_PREDICTOR = DepthAnythingV2Predictor(device="cpu", encoder="vits")


def get_initial_mask(
    img_path,
    tracking_points,
    trackings_input_label,
):
    yield from _get_initial_mask(
        img_path=img_path,
        tracking_points=tracking_points,
        trackings_input_label=trackings_input_label,
    )


# incrementally seen by the viewer. Also, there are no ephemeral RRDs to cleanup or manage.
@rr.thread_local_stream("sam2+depth_anything_v2")
def _get_initial_mask(
    img_path: Path,
    tracking_points: list | gr.State,
    trackings_input_label: list | gr.State,
):
    stream = rr.binary_stream()

    if img_path is None:
        raise gr.Error("Must provide an image to blur.")

    if isinstance(tracking_points, gr.State):
        tracking_points_list = tracking_points.value
    else:
        tracking_points_list = tracking_points

    if isinstance(trackings_input_label, gr.State):
        trackings_input_label_list = trackings_input_label.value
    else:
        trackings_input_label_list = trackings_input_label

    points: np.ndarray = np.array(tracking_points_list, dtype=np.float32)
    labels: np.ndarray = np.array(trackings_input_label_list, dtype=np.float32)

    parent_log_path = Path("world")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)
    rr.set_time_sequence("iteration", 0)

    blueprint = create_blueprint(parent_log_path)
    rr.send_blueprint(blueprint)

    img_dir = img_path.parent

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    DEPTH_PREDICTOR.set_model_device("cuda")

    with torch.inference_mode():
        inference_state = predictor.init_state(str(img_dir))

        frame_idx: int
        object_ids: list
        masks: Float32[torch.Tensor, "b 3 h w"]

        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state, frame_idx=0, obj_id=0, points=points, labels=labels
        )

        rgb = mmcv.imread(str(img_path), channel_order="rgb")
        depth_pred: RelativeDepthPrediction = DEPTH_PREDICTOR.__call__(
            rgb=rgb, K_33=None
        )
        masks: Bool[np.ndarray, "1 h w"] = (masks[0] > 0.0).numpy(force=True)
        log_relative_pred(
            parent_log_path=parent_log_path,
            relative_pred=depth_pred,
            rgb_hw3=rgb,
            seg_mask_hw=masks[0].astype(np.uint8),
            remove_flying_pixels=True,
        )

    yield stream.read(), inference_state


@no_type_check
def propagate_mask(
    img_path: Path,
    tracking_points: list | gr.State,
    trackings_input_label: list | gr.State,
    inference_state: dict | gr.State,
):
    yield from _propagate_mask(
        img_path=img_path,
        tracking_points=tracking_points,
        trackings_input_label=trackings_input_label,
        inference_state=inference_state,
    )


@rr.thread_local_stream("sam2+depth_anything_v2")
def _propagate_mask(
    img_path: Path,
    tracking_points: list | gr.State,
    trackings_input_label: list | gr.State,
    inference_state: dict | gr.State,
):
    stream = rr.binary_stream()

    if img_path is None:
        raise gr.Error("Must provide an image to blur.")

    if isinstance(tracking_points, gr.State):
        tracking_points_list = tracking_points.value
    else:
        tracking_points_list = tracking_points

    if isinstance(trackings_input_label, gr.State):
        trackings_input_label_list = trackings_input_label.value
    else:
        trackings_input_label_list = trackings_input_label

    if isinstance(inference_state, gr.State):
        inference_state = inference_state.value

    points: np.ndarray = np.array(tracking_points_list, dtype=np.float32)
    labels: np.ndarray = np.array(trackings_input_label_list, dtype=np.float32)

    parent_log_path = Path("world")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)
    rr.set_time_sequence("iteration", 0)

    blueprint = create_blueprint(parent_log_path)
    rr.send_blueprint(blueprint)

    img_dir = img_path.parent
    frames_path_list: list[Path] = sorted(img_dir.glob("*.jpg"), key=lambda p: p.name)

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    DEPTH_PREDICTOR.set_model_device("cuda")

    with torch.inference_mode():
        frame_idx: int
        object_ids: list
        masks: Float32[torch.Tensor, "b 3 h w"]

        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state, frame_idx=0, obj_id=0, points=points, labels=labels
        )

        # propagate the prompts to get masklets throughout the video
        for (frame_idx, object_ids, masks), frame_path in zip(
            predictor.propagate_in_video(inference_state), frames_path_list
        ):
            rr.set_time_sequence("frame", frame_idx)
            rgb = mmcv.imread(str(frame_path), channel_order="rgb")
            depth_pred: RelativeDepthPrediction = DEPTH_PREDICTOR.__call__(
                rgb=rgb, K_33=None
            )
            masks: Bool[np.ndarray, "1 h w"] = (masks[0] > 0.0).numpy(force=True)
            log_relative_pred(
                parent_log_path=parent_log_path,
                relative_pred=depth_pred,
                rgb_hw3=rgb,
                seg_mask_hw=masks[0].astype(np.uint8),
                remove_flying_pixels=True,
            )

            yield stream.read()


if IN_SPACES:
    propagate_mask = spaces.GPU(propagate_mask)
    get_initial_mask = spaces.GPU(get_initial_mask)


def get_point(
    point_type: Literal["include", "exclude"],
    tracking_points: gr.State | list,
    trackings_input_label: gr.State | list,
    input_first_frame_image: str | Path,
    evt: gr.SelectData,
):
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")

    # Convert the tracking_points and trackings_input_label to lists if they are gr.State objects
    if isinstance(tracking_points, gr.State):
        tracking_points_list: list = tracking_points.value
    else:
        tracking_points_list: list = tracking_points
    if isinstance(trackings_input_label, gr.State):
        trackings_input_label_list: list = trackings_input_label.value
    else:
        trackings_input_label_list: list = trackings_input_label

    tracking_points_list.append(evt.index)
    print(f"TRACKING POINT: {tracking_points}")

    if point_type == "include":
        trackings_input_label_list.append(1)
    elif point_type == "exclude":
        trackings_input_label_list.append(0)
    print(f"TRACKING INPUT LABEL: {trackings_input_label_list}")

    # Open the image and get its dimensions
    transparent_background: Image.Image = Image.open(input_first_frame_image).convert(
        "RGBA"
    )
    w, h = transparent_background.size

    # Define the circle radius as a fraction of the smaller dimension
    fraction: float = 0.02  # You can adjust this value as needed
    radius: int = int(fraction * min(w, h))

    # Create a transparent layer to draw on
    transparent_layer: UInt8[np.ndarray, "h w 4"] = np.zeros((h, w, 4), dtype=np.uint8)

    for index, track in enumerate(tracking_points_list):
        if trackings_input_label_list[index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)

    # Convert the transparent layer back to an image
    transparent_layer = Image.fromarray(transparent_layer, "RGBA")  # type: ignore
    selected_point_map = Image.alpha_composite(
        transparent_background,
        transparent_layer,  # type: ignore
    )

    return tracking_points, trackings_input_label, selected_point_map


def clear_points(image):
    # we clean all
    return [
        image,  # first_frame_path
        gr.State([]),  # tracking_points
        gr.State([]),  # trackings_input_label
        image,  # points_map
    ]


def rescale_img(
    img_hw3: UInt8[np.ndarray, "h w 3"], max_dim: int
) -> UInt8[np.ndarray, "..."]:
    # resize the image to have a max dim of 1024
    height, width, _ = img_hw3.shape
    current_dim = max(height, width)
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        img_hw3 = mmcv.imrescale(img=img_hw3, scale=scale_factor)
    return img_hw3


def preprocess_video(video_path: str) -> tuple[Path, gr.Accordion, gr.Image]:
    tmp_dir: str = tempfile.mkdtemp()
    print("Created temporary directory", tmp_dir)
    video: VideoReader = VideoReader(filename=str(video_path))

    target_fps: int = 10
    frame_interval: int = int(video.fps // target_fps)
    max_frames: int = 100
    total_saved_frames: int = 0
    max_size: int = 640

    for idx, bgr in enumerate(video):
        if idx % frame_interval == 0:
            if total_saved_frames >= max_frames:
                break
            bgr: np.ndarray = rescale_img(bgr, max_size)
            # 3. Save frames to temporary directory
            mmcv.imwrite(bgr, f"{tmp_dir}/{idx:05d}.jpg")
            total_saved_frames += 1

    first_frame_path: Path = Path(tmp_dir) / "00000.jpg"

    return (
        first_frame_path,
        gr.Accordion(open=False),
        gr.Image(value=str(first_frame_path), visible=True),
    )


with gr.Blocks() as demo:
    first_frame_path = gr.State()
    tracking_points = gr.State(value=[])
    trackings_input_label = gr.State(value=[])
    inference_state = gr.State()

    with gr.Tab("Streaming"):
        with gr.Row():
            with gr.Column():
                points_map = gr.Image(
                    interactive=False,
                    label="Point and Click Map",
                    visible=False,
                    type="filepath",
                    height=300,
                )
                with gr.Accordion("Your video IN", open=True) as video_in_drawer:
                    video_in = gr.Video(label="Video IN", format="mp4")

            with gr.Column():
                point_type = gr.Radio(
                    label="point type",
                    choices=["include", "exclude"],
                    value="include",
                    scale=1,
                )
                clear_points_btn = gr.Button("Clear Points", scale=1)
                get_initial_mask_btn = gr.Button("Get Initial Mask", scale=1)
                propagate_mask_btn = gr.Button("Propagate Mask", scale=1)

        with gr.Row():
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=800,
            )
        get_initial_mask_btn.click(
            get_initial_mask,
            inputs=[first_frame_path, tracking_points, trackings_input_label],
            outputs=[viewer, inference_state],
        )

        propagate_mask_btn.click(
            propagate_mask,
            inputs=[
                first_frame_path,
                tracking_points,
                trackings_input_label,
                inference_state,
            ],
            outputs=[viewer],
        )

        # triggered on video upload
        video_in.upload(
            fn=preprocess_video,
            inputs=[video_in],
            outputs=[first_frame_path, video_in_drawer, points_map],
        )

        # triggered when we click on image to add new points
        points_map.select(
            fn=get_point,
            inputs=[
                point_type,  # "include" or "exclude"
                tracking_points,  # get tracking_points values
                trackings_input_label,  # get tracking label values
                first_frame_path,  # gr.State() first frame path
            ],
            outputs=[
                tracking_points,  # updated with new points
                trackings_input_label,  # updated with corresponding labels
                points_map,  # updated image with points
            ],
            queue=False,
        )

        clear_points_btn.click(
            fn=clear_points,
            inputs=first_frame_path,  # we get the untouched hidden image
            outputs=[
                first_frame_path,
                tracking_points,
                trackings_input_label,
                points_map,
            ],
            queue=False,
        )
