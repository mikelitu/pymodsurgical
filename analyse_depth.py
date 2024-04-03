from video_reader import VideoReader
import depth
import torch
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_video(
    video_path: str | Path,
    metadata_path: str | Path,
    return_type: str = "numpy",
) -> VideoReader:
    with open(metadata_path, "r") as f:
        video_config = json.load(f)
    
    return VideoReader(video_path, video_config, return_type)


def init_depth(
    model_type: depth.ModelType = depth.ModelType.DPT_Large
) -> tuple[torch.nn.Module, torch.nn.Module]:
    return depth.load_depth_model_and_transform(model_type=model_type)


def plot_depth_map_with_frame(frame: np.ndarray, depth_map: np.ndarray, save_path: str | Path = None):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    
    ax1.imshow(frame)
    ax1.axis("off")
    ax1.set_title("Frame", fontsize=14)

    if len(depth_map.shape) == 3:
        depth_map = depth_map.squeeze(0)

    ax2.imshow(depth_map, cmap="plasma")
    ax2.axis("off")
    ax2.set_title("Depth map", fontsize=14)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
    
    plt.show()


def main():
    video_path = Path("videos/liver_stereo.avi")
    metadata_path = Path("videos/metadata.json")
    reader = read_video(video_path, metadata_path)
    intrinsics = reader.left_calibration_mat
    frames = reader.read()[0]
    # frames: list[np.ndarray] = [cv2.resize(frame, (40, 25)) for frame in frames]
    colors = [frame.reshape(-1, 3) / 255. for frame in frames]
    # frame = reader.read_frame(0)[0] if reader.video_type == "stereo" else reader.read_frame(0)
    depth_model, depth_transform = init_depth()
    depth_model.to(device)
    depth_frames = depth.calculate_depth_map_from_video(depth_model, depth_transform, frames, device)
    point_clouds = [depth.unproject_image_to_point_cloud(frame, intrinsics, False) for frame in depth_frames]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ani = animation.FuncAnimation(fig, depth.plot_point_cloud, init_func=depth.plot_point_cloud(point_clouds[0], colors[0], ax), frames=point_clouds, fargs=(colors[0], ax,), interval=1000, blit=False)
    ani.save("videos/point_cloud.mp4", writer="ffmpeg", fps=15)
    # unprojected_img = depth.unproject_image_to_point_cloud(depth_frame.squeeze(0), intrinsics, True)
    # print(unprojected_img.shape)
    # depth.plot_point_cloud(unprojected_img)


if __name__ == "__main__":
    main()
