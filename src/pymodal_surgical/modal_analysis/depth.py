import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from enum import StrEnum
from .math_helper import _norm_numpy, _norm_torch
from sklearn.cluster import KMeans

model_hub = "intel-isl/MiDaS"

class ModelType(StrEnum):
    DPT_Large = "DPT_Large"
    DPT_Hybrid = "DPT_Hybrid"
    MiDaS_small = "MiDaS_small"


def is_depth_img(depth: np.ndarray) -> bool:
    return depth.max() == 1.0 and depth.min() == 0.0


def norm_depth(depth: np.ndarray, as_img: bool = False) -> np.ndarray:
    if isinstance(depth, torch.Tensor):
        return _norm_torch(depth, as_img)
    return _norm_numpy(depth, as_img)


def load_depth_model_and_transform(model_type: ModelType = ModelType.DPT_Large) -> tuple[nn.Module, nn.Module]:
    model = torch.hub.load(model_hub, model_type)
    transform = torch.hub.load(model_hub, "transforms")
    if model_type == ModelType.DPT_Large or model_type == ModelType.DPT_Hybrid:
        transform = transform.dpt_transform
    else:
        transform = transform.small_transform
    return model, transform


def calculate_depth_map(
    model: nn.Module, 
    transform: nn.Module, 
    frame: np.ndarray | torch.Tensor,
    device: str = "cuda"
) -> np.ndarray:
    input_batch = transform(frame).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = F.interpolate(prediction.unsqueeze(1), size=(frame.shape[0], frame.shape[1]), mode="bicubic", align_corners=False).squeeze(1)
    output = prediction.cpu().numpy()
    return output


def calculate_depth_map_from_video(
    model: nn.Module,
    transform: nn.Module,
    video: list[np.ndarray],
    device: str = "cuda",
    batch_size: int = 10,
) -> np.ndarray:
    
    batch_size = min(batch_size, len(video))
    depth_map = []
    for i in range(0, len(video), batch_size):
        batch = np.array(video[i:i + batch_size])
        input_batch = torch.stack([transform(im).to(device) for im in batch]).squeeze(1).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = F.interpolate(prediction.unsqueeze(1), size=(batch[0].shape[0], batch[0].shape[1]), mode="bicubic", align_corners=False).squeeze(1)
        depth_map.append(prediction.cpu().numpy())
    return np.concatenate(depth_map, axis=0)


def apply_depth_culling(
    depth_map: np.ndarray, 
    frame: np.ndarray,
    near: float = 0.05,
    far: float = 0.95,
    invserse: bool = False
) -> torch.Tensor:
    
    culled_depth_map = np.where((depth_map >= near) & (depth_map <= far), depth_map, 0)
    culled_depth_map = (culled_depth_map - culled_depth_map.min()) / (culled_depth_map.max() - culled_depth_map.min())
    frame = (frame / 255).astype(np.float32)
    if invserse:
        culled_frame = frame * (1 - culled_depth_map[..., None])
    else:
        culled_frame = frame * culled_depth_map[..., None]
    return (255 * culled_frame).astype(np.uint8)


def create_rgbd(
    frame: torch.Tensor,
    depth_map: torch.Tensor
) -> torch.Tensor:
    if not is_depth_img(depth_map):
        depth_map = norm_depth(depth_map, False)
    
    if len(depth_map.shape) == 3:
        depth_map = depth_map.unsqueeze(1)
    
    if frame.shape[2:] != depth_map.shape[2:]:
        depth_map = F.interpolate(depth_map, (frame.shape[2], frame.shape[3]), mode="bicubic", align_corners=False)
    
    if frame.device != depth_map.device:
        depth_map = depth_map.to(frame.device)

    return torch.cat((frame, depth_map), dim=1)


def z_optical_flow(
    depth_map_1: np.ndarray, 
    depth_map_2: np.ndarray
) -> np.ndarray:
    """Simple approach to calculate the 'optical flow' from two depth maps"""
    if not is_depth_img(depth_map_1):
        depth_map_1 = norm_depth(depth_map_1)

    return depth_map_2 - depth_map_1


def z_optical_flow_from_video(
    depth_maps: list[np.ndarray]
) -> torch.Tensor:
    return torch.from_numpy(np.array([z_optical_flow(depth_maps[i], depth_maps[i + 1]) for i in range(len(depth_maps) - 1)])).unsqueeze(1)


def unproject_image_to_point_cloud(
    depth_map: np.ndarray, 
    K: np.ndarray,
    simplify: bool = False,
) -> np.ndarray:
    """Unproject a depth map to a point cloud"""
    inv_intrinsics = np.linalg.inv(K)
    height, width = depth_map.shape
    points = []
    for y in range(height):
        for x in range(width):
            Z = depth_map[y, x]
            if Z > 0:
                pixel = np.array([x, y, 1])
                point = Z * np.dot(inv_intrinsics, pixel)
                points.append(point)
    
    if simplify:
        points = simplify_point_cloud(np.array(points), 1000)

    return np.array(points)


def simplify_point_cloud(
    point_cloud: np.ndarray,
    num_samples: int
) -> np.ndarray:
    
    num_points_original = point_cloud.shape[0]
    if num_points_original <= num_samples:
        return point_cloud
    
    # Use KMeans clustering to find the representative points
    kmeans = KMeans(n_clusters=num_samples)
    kmeans.fit(point_cloud)
    return kmeans.cluster_centers_