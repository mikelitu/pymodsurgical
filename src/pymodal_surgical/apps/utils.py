import torch
import torchvision
import numpy as np
from pathlib import Path
from pymodal_surgical.video_processing.masking import Masking
from pymodal_surgical.video_processing.reader import VideoReader
from pymodal_surgical.modal_analysis import functions
from pymodal_surgical.modal_analysis.depth import load_depth_model_and_transform, ModelType, calculate_depth_map
from pymodal_surgical.modal_analysis import mode_shape_2_complex
from pymodal_surgical.modal_analysis.plot_utils import save_complex_mode_shape
from pymodal_surgical.modal_analysis.motion import get_motion_frequencies
from pymodal_surgical.modal_analysis.optical_flow import warp_flow
from PIL import Image
from torchvision.utils import flow_to_image
import cv2



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class ModeShapeCalculator():

    def __init__(
        self,
        config: dict,
    ) -> None:
        
        video_path = Path(config["video_path"])
        K = config["K"]
        experiment_name = video_path.stem
        self.experiment_dir = Path(f"results/{experiment_name}")
        self.cached = False

        if "batch_size" in config.keys():
            batch_size = config["batch_size"]
        else:
            batch_size = 100

        self._load_experiment(config)
        
        if not self.cached:
            if config["video_type"] == "stereo":
                self.mode_shapes, self.flows = self._calculate_mode_shapes(batch_size=batch_size, filter_config=config["filtering"], camera_pos="left")
            else:
                self.mode_shapes, self.flows = self._calculate_mode_shapes(batch_size=batch_size, filter_config=config["filtering"])
            
            self._save_complex_mode_shapes()
            self._save_rgb_mode_shapes()
            self._save_mode_target()
        
        self.complex_mode_shapes = mode_shape_2_complex(self.mode_shapes)
        self.frequencies = get_motion_frequencies(len(self.frames), K, 1./config["fps"])


    def _calculate_mode_shapes(
        self,
        batch_size: int = 100,
        filter_config: dict | None = None,
        camera_pos: str | None = None,
        save_flow_video: bool = False,
    ) -> torch.Tensor:
        
        return functions.calculate_mode_shapes(self.frames, self.K, None, batch_size, filter_config, self.mask, camera_pos, save_flow_video)
    

    def _load_experiment(
        self,
        config: dict,
    ) -> None:
        
        video_reader = VideoReader(config)

        self.frames = video_reader.read(int(config["start"]), int(config["end"]))

        if config["video_type"] == "stereo":
            self.frames = self.frames[0] # Just keep the left frames

        if self.experiment_dir.exists():
            print(f"Experiment directory {self.experiment_dir} already exists. Trying to load cached data.")
            mode_shape_dir = self.experiment_dir/"mode_shapes"
            if mode_shape_dir.exists():
                print(f"Loading cached mode shapes from {mode_shape_dir}")
                mode_shapes = torch.zeros((len(list(mode_shape_dir.glob("*.png"))), 4, self.frames[0].shape[0], self.frames[0].shape[1]))
                for file in sorted(mode_shape_dir.glob("*.png")):
                    mode_n = file.stem.split("_")[1]
                    img_mode_shape = Image.open(file)
                    tensor_mode_shape = torch.tensor(np.array(img_mode_shape)).permute(2, 0, 1)
                    mode_shapes[int(mode_n)] = tensor_mode_shape
                self.cached = True
                self.mode_shapes = mode_shapes
                return
        
        
        self.depth_model, self.depth_transform = load_depth_model_and_transform(ModelType.DPT_Large, device)
        if config["masking"]["enabled"]:
            self.mask = Masking(config["masking"]["mask"], video_reader.video_type)
        else:
            self.mask = None
        
        self.K = config["K"]


    def _calculate_depth(
        self
    ) -> torch.Tensor:
        pass
    

    def _save_complex_mode_shapes(
        self
    ) -> None:
        
        save_path = self.experiment_dir/"mode_shapes"
        save_complex_mode_shape(self.mode_shapes, Path(save_path))


    def _save_rgb_mode_shapes(
        self
    ) -> None:
        
        """
        Save the RGB mode shapes. We follow the same technique as in the pymodal_surgical.modal_analysis.plot_utils.save_complex_mode_shape.
        We use the utils from torchvision.utils.flow_to_image to convert the complex mode shapes to RGB images.
        """
        save_path = self.experiment_dir/"rgb_mode_shapes"
        save_path.mkdir(parents=True, exist_ok=True)
        complex_mode_shape = mode_shape_2_complex(self.mode_shapes)
        abs_mode_shapes = torch.abs(complex_mode_shape)
        for i in range(abs_mode_shapes.shape[0]):
            img_mode_shape = flow_to_image(abs_mode_shapes[i].cpu())
            img_mode_shape = torchvision.transforms.functional.to_pil_image(img_mode_shape)
            img_mode_shape.save(save_path/f"mode_{i}.png")


    def _save_mode_target(
        self
    ) -> None:
        
        """
        Save the mode response of the reference frame.
        """

        save_path = self.experiment_dir/"mode_target"
        save_path.mkdir(parents=True, exist_ok=True)
        complex_mode_shape = mode_shape_2_complex(self.mode_shapes)

        # Get the reference frame
        frame = self.frames[0]
        abs_mode_shapes = torch.abs(complex_mode_shape)
        tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).to(device)
        depth_map = calculate_depth_map(self.depth_model, self.depth_transform, frame, device=device)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        for i in range(abs_mode_shapes.shape[0]):
            increased_mode_shape = abs_mode_shapes[i] * 10.
            increased_mode_shape = increased_mode_shape.to(device)
            target_mode_shape = warp_flow(tensor_frame, increased_mode_shape, depth_map=depth_map, mask=self.mask.mask.cpu().numpy(), near=0.0, far=1.0)
            img_mode_shape = Image.fromarray(target_mode_shape)
            img_mode_shape.save(save_path/f"mode_{i}.png")