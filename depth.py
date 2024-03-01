import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from enum import StrEnum

class ModelType(StrEnum):
    DPT_Large = "DPT_Large"
    DPT_Hybrid = "DPT_Hybrid"
    MiDaS_small = "MiDaS_small"

model_hub = "intel-isl/MiDaS"

def load_depth_model_and_transform(model_type: ModelType) -> tuple[nn.Module, nn.Module]:
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
    frame: np.ndarray | torch.Tensor
) -> np.ndarray:
    input_batch = transform(frame).to("cuda")
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = F.interpolate(prediction.unsqueeze(1), size=(frame.shape[0], frame.shape[1]), mode="bicubic", align_corners=False).squeeze(1)
    output = prediction.cpu().numpy()
    return output


def apply_depth_culling(
    depth_map: np.ndarray, 
    frame: np.ndarray,
    near: float = 0.05,
    far: float = 0.95
) -> torch.Tensor:
    
    culled_depth_map = np.where((depth_map >= near) & (depth_map <= far), depth_map, 0)
    culled_depth_map = (culled_depth_map - culled_depth_map.min()) / (culled_depth_map.max() - culled_depth_map.min())
    frame = (frame / 255).astype(np.float32)
    culled_frame = frame * culled_depth_map[..., None]
    return (255 * culled_frame).astype(np.uint8)
