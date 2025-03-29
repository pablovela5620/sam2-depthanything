from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, TypedDict

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import torch
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8, UInt16
from monopriors.dc_utils import read_video_frames
from monopriors.depth_utils import depth_edges_mask
from monopriors.relative_depth_models.depth_anything_v2 import (
    DepthAnythingV2Predictor,
    RelativeDepthPrediction,
)
from monopriors.relative_depth_models.video_depth_anything import VideoDepthAnythingPredictor
from numpy import ndarray
from serde import from_dict
from serde.json import to_json
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters, rescale_intri
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from simplecv.video_io import MultiVideoReader
from torch import Tensor
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sam2_depthanything.vggt_utils import VGGTPredictions, create_blueprint, preprocess_images


@dataclass
class ProcessConfig:
    rr_config: RerunTyroConfig
    video_dir: Path = Path("/mnt/12tbdrive/data/HO-cap/sample/subject_8/20231024_180733/raw_videos/")
    device: Literal["cpu", "cuda"] = "cuda"
    confidence_threshold: float = 50.0


@dataclass
class CalibrationData:
    cam_name: str
    image: Float32[ndarray, "H W 3"]
    depth_map: UInt16[ndarray, "H W"]
    confidence_mask: UInt8[ndarray, "H W"]
    pointcloud: o3d.geometry.PointCloud
    pinhole_param: PinholeParameters


def compute_scale_and_shift_full(
    prediction: Float32[ndarray, "H W"], target: Float32[ndarray, "H W"], mask: Bool[ndarray, "H W"]
) -> tuple[float, float]:
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return float(x_0), float(x_1)


def generate_camera_parameters(
    pred_class: VGGTPredictions,
    img_tensors: Float32[Tensor, "num_img 3 resized_h resized_w"],
    rgb_list: list[UInt8[ndarray, "original_h original_w 3"]],
    confidence_threshold: float,
) -> list[CalibrationData]:
    pred_class = pred_class.remove_batch_dim_if_one()

    # Generate world points from depth map,this is usually more accurate than the world points from pose encoding
    depth_maps: Float32[ndarray, "num_cams resized_h resized_w 1"] = pred_class.depth
    world_points: Float32[ndarray, "num_cams resized_h resized_w 3"] = unproject_depth_map_to_point_map(
        depth_maps, pred_class.cam_T_world, pred_class.intrinsic
    ).astype(np.float32)

    # Get colors from original images and reshape them to match points
    processed_imgs: Float32[ndarray, "num_cams 3 resized_h resized_w"] = img_tensors.numpy(force=True)
    # Rearrange to match point shape expectation
    processed_imgs: Float32[ndarray, "num_cams resized_h resized_w 3"] = rearrange(
        processed_imgs,
        "num_cams C resized_h resized_w -> num_cams resized_h resized_w C",
    )
    # Flatten both points and colors
    flattened_points: Float32[ndarray, "num_points 3"] = rearrange(
        world_points,
        "num_cams resized_h resized_w C -> (num_cams resized_h resized_w) C",
    )
    flattened_colors: Float32[ndarray, "num_points 3"] = rearrange(
        processed_imgs,
        "num_cams resized_h resized_w C -> (num_cams resized_h resized_w) C",
    )

    depth_confs: Float32[ndarray, "num_cams resized_h resized_w"] = pred_class.depth_conf
    conf: Float32[ndarray, "num_points"] = depth_confs.reshape(-1)  # noqa UP037

    # Convert percentage threshold to actual confidence value
    conf_threshold = 0.0 if confidence_threshold == 0.0 else np.percentile(conf, confidence_threshold)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d: Float32[ndarray, "num_points 3"] = flattened_points[conf_mask]
    colors_rgb: Float32[ndarray, "num_points 3"] = flattened_colors[conf_mask]

    # Create an empty point cloud
    pcd = o3d.geometry.PointCloud()

    # Ensure your positions and colors are of the appropriate type (typically float64 for points)
    pcd.points = o3d.utility.Vector3dVector(vertices_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    calib_data_list: list[CalibrationData] = []
    for idx, (intri, extri, processed_img, original_img, depth_map, depth_conf) in enumerate(
        zip(
            pred_class.intrinsic,
            pred_class.cam_T_world,
            processed_imgs,
            rgb_list,
            depth_maps,
            depth_confs,
            strict=True,
        )
    ):
        cam_name: str = f"camera_{idx}"
        intri_param = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(intri[0, 0]),
            fl_y=float(intri[1, 1]),
            cx=float(intri[0, 2]),
            cy=float(intri[1, 2]),
            width=processed_img.shape[1],
            height=processed_img.shape[0],
        )
        extri_param = Extrinsics(
            cam_R_world=extri[:, :3],
            cam_t_world=extri[:, 3],
        )
        pinhole_param = PinholeParameters(name=cam_name, intrinsics=intri_param, extrinsics=extri_param)
        conf_threshold = 0.0 if confidence_threshold == 0.0 else np.percentile(depth_conf, confidence_threshold)
        conf_mask = (depth_conf >= conf_threshold) & (depth_conf > 1e-5)
        # filter depth map based on confidence
        depth_map = depth_map.squeeze()
        depth_map[~conf_mask] = 0.0
        # resize image, confidence mask and depth map to original image size
        # Use INTER_LINEAR for the processed RGB image (standard for color images)
        processed_img = cv2.resize(
            processed_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        # Use INTER_NEAREST for the confidence mask to preserve binary values
        conf_mask = cv2.resize(
            conf_mask.astype(np.float32),
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        # Use INTER_NEAREST for depth map to preserve discontinuities and avoid floating artifacts
        depth_map = cv2.resize(
            depth_map, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        # rescale camera parameters to original image size
        pinhole_param.intrinsics = rescale_intri(
            pinhole_param.intrinsics,
            target_width=original_img.shape[1],
            target_height=original_img.shape[0],
        )
        calib_data_list.append(
            CalibrationData(
                cam_name=cam_name,
                image=processed_img,
                depth_map=(depth_map * 1000).astype(np.uint16),  # convert to uint16
                confidence_mask=(conf_mask * 255).astype(np.uint8),
                pointcloud=pcd,
                pinhole_param=pinhole_param,
            )
        )

    return calib_data_list


def save_caliberation_data(data_dir: Path, sequence_name: str, calibration_data: list[CalibrationData]):
    sequence_dir: Path = data_dir / sequence_name
    sequence_dir.mkdir(parents=True, exist_ok=True)
    # Create directories for videos, depth maps, and confidence maps
    videos_dir = sequence_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = sequence_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = sequence_dir / "confidences"
    conf_dir.mkdir(parents=True, exist_ok=True)

    pinhole_parameters: list[PinholeParameters] = [calib_data.pinhole_param for calib_data in calibration_data]
    # Save camera parameters to a JSON file
    camera_parameters_path = sequence_dir / "camera_parameters.json"
    camera_parameters_json: str = to_json(pinhole_parameters)
    print("Camera parameters JSON:", camera_parameters_json)
    # Save the JSON string to a file
    with open(camera_parameters_path, "w") as f:
        f.write(camera_parameters_json)
    # Save point cloud to a PLY file
    point_cloud_path = sequence_dir / "point_cloud.ply"
    o3d.io.write_point_cloud(str(point_cloud_path), calibration_data[0].pointcloud)

    for calib_data in tqdm(calibration_data, desc="Saving calibration data", unit="camera"):
        cam_name: str = calib_data.cam_name
        # Create camera specific directory
        (depth_dir / cam_name).mkdir(parents=True, exist_ok=True)
        depth_path = depth_dir / cam_name / "depth.png"
        depth_map: UInt16[ndarray, "H W"] = calib_data.depth_map
        # Save depth map as PNG
        cv2.imwrite(str(depth_path), depth_map)
        (conf_dir / cam_name).mkdir(parents=True, exist_ok=True)
        conf_path = conf_dir / cam_name / "conf.jpg"
        cv2.imwrite(str(conf_path), calib_data.confidence_mask.astype(np.uint8))


def process_data(config: ProcessConfig):
    parent_log_path: Path = Path("world")
    video_paths = sorted(config.video_dir.glob("*.mp4"))
    assert len(video_paths) > 0, f"No videos found in {config.video_dir}"

    mv_reader = MultiVideoReader(video_paths=video_paths)

    blueprint = create_blueprint(parent_log_path=parent_log_path, image_paths=video_paths)
    rr.send_blueprint(blueprint)

    load_start: float = timer()
    print("Loading model...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(config.device)
    print("Model loaded in", timer() - load_start, "seconds")

    rr.set_time_sequence("timeline", 0)

    bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[0]
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

    img_tensors: Float32[Tensor, "num_img 3 H W"] = preprocess_images(rgb_list).to(config.device)
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # Run inference
    print("Running inference...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        # run model and convert to dataclass for type validaton + easy access
        predictions: dict = vggt_model(img_tensors)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], img_tensors.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].numpy(force=True)

    # Convert from dict to dataclass and performs runtime type validation for easy access
    pred_class: VGGTPredictions = from_dict(VGGTPredictions, predictions)
    calibration_data: list[CalibrationData] = generate_camera_parameters(
        pred_class,
        img_tensors=img_tensors,
        rgb_list=rgb_list,
        confidence_threshold=config.confidence_threshold,
    )

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            calibration_data[0].pointcloud.points,
            colors=calibration_data[0].pointcloud.colors,
        ),
    )
    calib_data: CalibrationData
    for calib_data in calibration_data:
        cam_log_path: Path = parent_log_path / calib_data.cam_name

        mask: Float32[ndarray, "H W"] = calib_data.confidence_mask.astype(np.float32)
        depth_map: Float32[ndarray, "H W"] = (calib_data.depth_map / 1000).astype(np.float32)

        log_pinhole(
            calib_data.pinhole_param,
            cam_log_path=cam_log_path,
            image_plane_distance=0.1,
            static=True,
        )

        rr.log(f"{cam_log_path}/pinhole/image", rr.Image(calib_data.image), static=True)
        rr.log(
            f"{cam_log_path}/pinhole/confidence",
            rr.Image(mask),
            static=True,
        )
        rr.log(
            f"{cam_log_path}/pinhole/depth",
            rr.DepthImage(depth_map, draw_order=1),
            static=True,
        )

    # save the calibration data
    save_caliberation_data(data_dir=Path("data/example_data"), sequence_name="0", calibration_data=calibration_data)

    # Clean up
    torch.cuda.empty_cache()
    del vggt_model

    model: Literal["da2", "vda"] = "da2"

    start = timer()
    print("Generating Depth Maps...")
    if model == "da2":
        DEPTH_PREDICTOR = DepthAnythingV2Predictor(device="cpu", encoder="vits")
        DEPTH_PREDICTOR.set_model_device("cuda")

        # Define a typed dictionary for camera scale and shift
        class CameraScaleShift(TypedDict):
            scale: float
            shift: float

        # Store scale and shift for each camera
        camera_scale_shift: dict[int, CameraScaleShift] = {}

        # propagate the prompts to get masklets throughout the video
        for frame_idx, bgr_list in tqdm(enumerate(mv_reader), desc="Processing frames", total=len(mv_reader)):
            rr.set_time_sequence("frame", frame_idx)
            rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

            for cam_idx, (rgb, calib_data) in enumerate(zip(rgb_list, calibration_data, strict=True)):
                cam_log_path: Path = parent_log_path / calib_data.cam_name / "pinhole"
                pinhole_param: PinholeParameters = calib_data.pinhole_param
                depth_pred: RelativeDepthPrediction = DEPTH_PREDICTOR.__call__(
                    rgb=rgb, K_33=pinhole_param.intrinsics.k_matrix.astype(np.float32)
                )

                mono_disparity: Float32[np.ndarray, "h w"] = depth_pred.depth
                mask = calib_data.confidence_mask.astype(np.bool_)
                sparse_depth: Float32[ndarray, "H W"] = (calib_data.depth_map / 1000).astype(np.float32)

                # Calculate scale and shift only on the first frame
                if frame_idx == 0:
                    scale, shift = compute_scale_and_shift_full(
                        prediction=mono_disparity.astype(np.float32),
                        target=sparse_depth,
                        mask=mask,
                    )
                    camera_scale_shift[cam_idx] = {"scale": scale, "shift": shift}
                else:
                    # Reuse scale and shift from first frame
                    scale: float = camera_scale_shift[cam_idx]["scale"]
                    shift: float = camera_scale_shift[cam_idx]["shift"]

                aligned_depth: Float32[np.ndarray, "h w"] = mono_disparity.astype(np.float32) * scale + shift
                aligned_depth[aligned_depth < 0] = 0
                # mask out the pixels that are not in the original depth map
                aligned_depth[aligned_depth > np.max(sparse_depth)] = 0  # Clip values above max sparse depth

                edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(aligned_depth, threshold=0.01)
                aligned_depth: Float32[np.ndarray, "h w"] = aligned_depth * ~edges_mask

                # log to cam_log_path to avoid backprojecting disparity
                rr.log(f"{cam_log_path}/aligned_depth", rr.DepthImage(aligned_depth))

    elif model == "vda":
        DEPTH_PREDICTOR = VideoDepthAnythingPredictor(device="cuda", encoder="vits")

        # Define a typed dictionary for camera scale and shift
        class CameraScaleShift(TypedDict):
            scale: float
            shift: float

        # Store scale and shift for each camera
        camera_scale_shift: dict[int, CameraScaleShift] = {}

        # instead of iterating over the frames, we will iterate over the video reader
        for cam_idx, (video_path, calib_data) in enumerate(
            tqdm(
                zip(mv_reader.video_paths, calibration_data, strict=True),
                desc="Processing videos",
                total=len(calibration_data),
            )
        ):
            cam_log_path: Path = parent_log_path / calib_data.cam_name / "pinhole"
            read_output: tuple[UInt8[ndarray, "T H W 3"], float] = read_video_frames(
                video_path, process_length=-1, target_fps=-1, max_res=-1
            )
            frames: UInt8[ndarray, "T H W 3"] = read_output[0]
            depths: list[RelativeDepthPrediction] = DEPTH_PREDICTOR(
                frames, K_33=calib_data.pinhole_param.intrinsics.k_matrix.astype(np.float32)
            )
            for frame_idx, depth_pred in enumerate(depths):
                rr.set_time_sequence("frame", frame_idx)
                mono_disparity: Float32[np.ndarray, "h w"] = depth_pred.depth
                mask = calib_data.confidence_mask.astype(np.bool_)
                sparse_depth: Float32[ndarray, "H W"] = (calib_data.depth_map / 1000).astype(np.float32)

                # Calculate scale and shift only on the first frame
                if frame_idx == 0:
                    scale, shift = compute_scale_and_shift_full(
                        prediction=mono_disparity.astype(np.float32),
                        target=sparse_depth,
                        mask=mask,
                    )
                    camera_scale_shift[cam_idx] = {"scale": scale, "shift": shift}
                else:
                    # Reuse scale and shift from first frame
                    scale: float = camera_scale_shift[cam_idx]["scale"]
                    shift: float = camera_scale_shift[cam_idx]["shift"]

                aligned_depth: Float32[np.ndarray, "h w"] = mono_disparity.astype(np.float32) * scale + shift
                aligned_depth[aligned_depth < 0] = 0
                # mask out the pixels that are not in the original depth map
                aligned_depth[aligned_depth > np.max(sparse_depth)] = 0  # Clip values above max sparse depth

                edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(aligned_depth, threshold=0.01)
                aligned_depth: Float32[np.ndarray, "h w"] = aligned_depth * ~edges_mask

                # log to cam_log_path to avoid backprojecting disparity
                rr.log(f"{cam_log_path}/aligned_depth", rr.DepthImage(aligned_depth))
    print(f"Depth Maps from {model} generated in {timer() - start:.2f} seconds")
