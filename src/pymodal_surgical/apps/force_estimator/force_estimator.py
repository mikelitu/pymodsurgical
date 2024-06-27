import torch
import numpy as np
from pathlib import Path
from pymodal_surgical.apps.utils import ModeShapeCalculator
from pymodal_surgical.video_processing.reader import VideoReader
from pymodal_surgical.modal_analysis.depth import load_depth_model_and_transform, calculate_depth_map, ModelType
import pymodal_surgical.modal_analysis.optical_flow as optical_flow
import pymodal_surgical.modal_analysis.force as force
import pymodal_surgical.modal_analysis.math_helper as math_helper
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy import signal
from pymodal_surgical.utils import create_save_dir
from torchvision.utils import flow_to_image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pymodal_surgical.video_processing.filtering import GaussianFiltering

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class ForceEstimator():

    """
    A class to estimate and visualize forces from video data using optical flow and mode shapes.

    Attributes:
        mode_shapes (torch.Tensor): The complex mode shapes calculated from the mode shape configuration.
        frequencies (torch.Tensor): The frequencies corresponding to the mode shapes.
        fps (int): Frames per second of the video.
        filter (GaussianFiltering): Gaussian filter applied to the forces.
        save_force (bool): Flag indicating whether to save the estimated forces.
        force_save_path (str): Path to save the force video.
        force_video_writer (cv2.VideoWriter): Writer object for saving force video.
        force_mask (torch.Tensor): Mask for blending force with frames.
        pixels (tuple): Region of interest for force estimation.
        force_video_reader (VideoReader): Reader object for the input video.
        flow_model: Model for estimating optical flow.
    """
    
    def __init__(
        self,
        force_estimation_config: dict,
        mode_shape_config: dict,
    ) -> None:
        
        """
        Initialize the ForceEstimator.

        Args:
            force_estimation_config (dict): Configuration for force estimation.
            mode_shape_config (dict): Configuration for mode shape calculation.
        """
        mode_shape_calculator = ModeShapeCalculator(mode_shape_config)

        self.mode_shapes = mode_shape_calculator.complex_mode_shapes
        self.frequencies = mode_shape_calculator.frequencies
        self.fps = mode_shape_calculator.fps
        self._load_force_video(force_estimation_config)

        self.filter = GaussianFiltering(15, 3.0)


        if "save_force" in force_estimation_config.keys() and force_estimation_config["save_force"] != "":
            self.save_force = True
            self.force_save_path = force_estimation_config["save_force"]
            force_video_path = create_save_dir(self.force_save_path, Path(mode_shape_config["video_path"]).stem)
            video_name = Path(force_estimation_config["video_path"]).stem + ".mp4"
            force_video_file = force_video_path /"force"/ video_name
            force_video_file.parent.mkdir(parents=True, exist_ok=True)
            if self.force_video_reader.video_type == "stereo":
                width, height = self.force_video_reader.video_width // 2, self.force_video_reader.video_height
            else:
                width, height = self.force_video_reader.video_width, self.force_video_reader.video_height
            
            self.force_video_writer = cv2.VideoWriter(str(force_video_file), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (height, width))
            self._create_mask((height, width))
        
        else:
            self.save_force = False
            self.force_save_path = None

    
    def _load_force_video(
        self,
        force_estimation_config: dict
    ) -> None:
        """
        Load the force video based on the given configuration.

        Args:
            force_estimation_config (dict): Configuration for force estimation.
        """
        self.force_video_reader = VideoReader(video_config=force_estimation_config)
        self.flow_model = optical_flow.load_flow_model(device)
        
        if "pixels" in force_estimation_config.keys() and force_estimation_config["pixels"] is not None:
            self.pixels = force_estimation_config["pixels"]
        else:
            self.pixels = None


    def _create_mask(
        self,
        size: tuple[int, int] = (128, 128)
    ) -> None:
        """
        Create a mask for blending forces with video frames.

        Args:
            size (tuple[int, int]): Size of the mask (height, width). Default is (128, 128).
        """
        self.force_mask = torch.zeros((2, *size), dtype=torch.float32)
    

    def _blend_frame_with_force(
        self,
        frame: np.ndarray,
        force: torch.Tensor,
    ) -> np.ndarray:
        """
        Blend a video frame with the estimated force.

        Args:
            frame (np.ndarray): The video frame.
            force (torch.Tensor): The estimated force.

        Returns:
            np.ndarray: The blended frame.
        """
        self.force_mask[:, self.pixels[0][0]:self.pixels[0][1], self.pixels[1][0]:self.pixels[1][1]] = force.transpose(2, 1)
        force_mask = flow_to_image(self.force_mask)
        force_mask = force_mask.permute(1, 2, 0).numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.addWeighted(frame, 0.7, force_mask, 0.3, 0)
        return frame


    def _blend_quiver_with_frame(
        self,
        frame: np.ndarray,
        force: torch.Tensor,
    ) -> np.ndarray:
        """
        Blend a video frame with a quiver plot of the estimated force.

        Args:
            frame (np.ndarray): The video frame.
            force (torch.Tensor): The estimated force.

        Returns:
            np.ndarray: The blended frame with a quiver plot.
        """
        gd = 20
        Y, X = np.mgrid[0:frame.shape[0]:gd, 0:frame.shape[1]:gd]
        self.force_mask[:, self.pixels[0][0]:self.pixels[0][1], self.pixels[1][0]:self.pixels[1][1]] = force
        V = self.force_mask[0, ::gd, ::gd].numpy()
        U = self.force_mask[1, ::gd, ::gd].numpy()

        dpi = 100
        figsize = (frame.shape[0] / dpi, frame.shape[1] / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot([0, 0, 1, 1])
        ax.imshow(frame, zorder=0, alpha=1.0, interpolation="hermite")
        rec_limits = (self.pixels[0][0], self.pixels[0][1], self.pixels[1][0], self.pixels[1][1])
        rect = patches.Rectangle((rec_limits[0], rec_limits[2]), rec_limits[1] - rec_limits[0], rec_limits[3] - rec_limits[2], linewidth=1.0, edgecolor="k", facecolor="none")
        ax.quiver(Y, X, V, U, color="yellow", scale=25)
        ax.add_patch(rect)
        ax.axis("off")

        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(int(height), int(width), 3)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Release the figure resources (important to avoid memory issues)
        plt.close(fig)

        return image


    def _show_frames(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        Display a video frame.

        Args:
            frame (np.ndarray): The video frame.
        """
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)


    def _select_roi(
        self,
        frame: np.ndarray,
    ) -> tuple[int, int]:
        """
        Select a region of interest (ROI) from the video frame.

        Args:
            frame (np.ndarray): The video frame.

        Returns:
            tuple[int, int]: The selected ROI as (x, y, width, height).
        """
        frame = cv2.resize(frame, (self.mode_shapes.shape[3], self.mode_shapes.shape[2]))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bbox = cv2.selectROI("Frame", frame, fromCenter=True)
        cv2.destroyAllWindows()
        return bbox
    
    def calculate_force(
        self,
        idx_1: int,
        idx_2: int,
        simplify_force: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the force between two video frames.

        Args:
            idx_1 (int): Index of the first frame.
            idx_2 (int): Index of the second frame.
            simplify_force (bool): Flag indicating whether to simplify the force calculation. Default is False.

        Returns:
            torch.Tensor: The estimated force.
        """
        # Load the frames
        if self.force_video_reader.video_type == "stereo":
            frame_1 = self.force_video_reader.read_frame(idx_1)[0]
            frame_2 = self.force_video_reader.read_frame(idx_2)[0]
        
        else:
            frame_1 = self.force_video_reader.read_frame(idx_1)
            frame_2 = self.force_video_reader.read_frame(idx_2)

        if self.pixels is None:
            roi = self._select_roi(frame_1)
            pix_w = (roi[1], roi[1] + roi[3])
            pix_h = (roi[0], roi[0] + roi[2])
            self.pixels = (pix_h, pix_w)
            print(f"Selected ROI: {self.pixels}")

        # Preprocess the frames for the calculation of optical flow
        processed_frame_1 = optical_flow.preprocess_for_raft(frame_1).unsqueeze(0).to(device)
        processed_frame_2 = optical_flow.preprocess_for_raft(frame_2).unsqueeze(0).to(device)

        # Calculate the optical flow
        flow = optical_flow.estimate_flow(self.flow_model, processed_frame_1, processed_frame_2)
        flow = flow.squeeze(0).detach().cpu()

        # The optical flow corresponds to the desired displacement of the pixels
        # We need to calculate the force that is applied to the pixels
        # We can do this by calculating the constraint force
        # The constraint force is the force that is applied to the pixels to keep them in place

        # Calculate the constraint force
        modal_coordinates = torch.zeros(self.mode_shapes.shape[0], 2)
        norm_mode_shapes = math_helper.orthonormal_normalization(self.mode_shapes)
        constraint_force = force.calculate_force_from_displacement_map(norm_mode_shapes, flow, modal_coordinates, self.frequencies, self.pixels, simplify_force=simplify_force, timestep=0.03)
        
        if self.save_force:

            if simplify_force:
                raise ValueError("Cannot save simplified force to plot as a video grid")

            blended_force = self._blend_quiver_with_frame(frame_2, constraint_force)
            self.force_video_writer.write(blended_force)

        return constraint_force

