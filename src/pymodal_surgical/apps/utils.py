import torch
import numpy as np
from pathlib import Path
from pymodal_surgical.video_processing.masking import Masking
from pymodal_surgical.video_processing.reader import VideoReader
from pymodal_surgical.modal_analysis import functions
from pymodal_surgical.modal_analysis.depth import load_depth_model_and_transform, ModelType
from pymodal_surgical.modal_analysis import mode_shape_2_complex
from pymodal_surgical.modal_analysis.plot_utils import save_complex_mode_shape
from pymodal_surgical.modal_analysis.motion import get_motion_frequencies
from PIL import Image


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class ModeShapeCalculator():

    def __init__(
        self,
        video_path: str | Path,
        config: dict,
        K: int = 16,
    ) -> None:
        
        experiment_name = video_path.stem
        self.experiment_dir = Path(f"results/{experiment_name}")
        self.cached = False

        self._load_experiment(video_path, config, K)
        
        if not self.cached:
            self.mode_shapes, self.flows = self._calculate_mode_shapes(filter_config=config["filtering"])
            self._save_complex_mode_shapes()
        
        self.complex_mode_shapes = mode_shape_2_complex(self.mode_shapes)
        self.frequencies = get_motion_frequencies(len(self.frames), K, 1./config["video_config"]["fps"])

    def _calculate_mode_shapes(
        self,
        depth_maps: list[np.ndarray] | None = None,
        batch_size: int = 500,
        filter_config: dict | None = None,
        camera_pos: str | None = None,
        save_flow_video: bool = False,
    ) -> torch.Tensor:
        
        return functions.calculate_mode_shapes(self.frames, self.K, None, batch_size, filter_config, self.mask, camera_pos, save_flow_video)
    
    def _load_experiment(
        self,
        video_path: str | Path,
        config: dict,
        K: int,
    ) -> None:
        
        video_reader = VideoReader(video_path, video_config=config["video_config"])
        self.frames = video_reader.read(config["start"], config["end"])

        if self.experiment_dir.exists():
            print(f"Experiment directory {self.experiment_dir} already exists. Trying to load cached data.")
            mode_shape_dir = self.experiment_dir/"mode_shapes"
            if mode_shape_dir.exists():
                print(f"Loading cached mode shapes from {mode_shape_dir}")
                mode_shapes = torch.zeros((len(list(mode_shape_dir.glob("*.png"))), 4, 128, 128))
                for file in sorted(mode_shape_dir.glob("*.png")):
                    mode_n = file.stem.split("_")[1]
                    img_mode_shape = Image.open(file)
                    tensor_mode_shape = torch.tensor(np.array(img_mode_shape)).permute(2, 0, 1)
                    mode_shapes[int(mode_n)] = tensor_mode_shape
                self.cached = True
                self.mode_shapes = mode_shapes
                return
        
        
        self.depth_model, self.depth_transform = load_depth_model_and_transform(ModelType.DPT_Large)
        if config["masking"]["enabled"]:
            self.mask = Masking(config["masking"]["mask"], video_reader.video_type)
        else:
            self.mask = None
        
        self.K = K

    def _calculate_depth(
        self
    ) -> torch.Tensor:
        pass

    def _save_complex_mode_shapes(
        self
    ) -> None:
        
        save_path = self.experiment_dir/"mode_shapes"
        save_complex_mode_shape(self.mode_shapes, Path(save_path))