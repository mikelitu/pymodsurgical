import numpy as np
import cv2
import torch

class GaussianFiltering:

    def __init__(self, kernel_size: tuple[int, int] | int = 15, sigma: float | tuple[float, float] = 0):
        self.sigma = sigma
        if isinstance(kernel_size, int):
            if kernel_size % 2 == 0:
                raise ValueError("Kernel size must be an odd number")
            self.kernel_size = (kernel_size, kernel_size)
        else:
            if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                raise ValueError("Kernel size must be an odd number")
            self.kernel_size = kernel_size
    

    def _compute_local_contrast(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            if img.shape[2] == 2:
                img = np.concatenate((img, np.zeros_like(img[:, :, 0:1])), axis=2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        local_std = cv2.GaussianBlur(img.astype(np.float32) ** 2, self.kernel_size, self.sigma) - cv2.GaussianBlur(img.astype(np.float32), self.kernel_size, self.sigma) ** 2
        local_std = np.sqrt(np.maximum(local_std, 0))
        local_std = local_std / local_std.max()
        return local_std
    
    def _blur_image(self, img: np.ndarray) -> np.ndarray:
        if img.shape[2] == 2:
            img = np.concatenate((img, np.zeros_like(img[:, :, 0:1])), axis=2)
            changed = True
        else:
            changed = False
        blurred_img = cv2.GaussianBlur(img, self.kernel_size, self.sigma)

        return blurred_img[:, :, :2] if changed else blurred_img
    
    def _filter_image(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        local_std_img = self._compute_local_contrast(img)
        weighted_flow = flow * local_std_img[..., np.newaxis]
        # Normalize the weighted flow
        weighted_flow = (weighted_flow - weighted_flow.min()) / (weighted_flow.max() - weighted_flow.min() + 1e-10)
        blurred_weighted_flow = self._blur_image(weighted_flow)
        blurred_flow = self._blur_image(flow)
        # Normalize the blurred flow by the blurred weighted flow
        blurred_flow = blurred_flow / (blurred_weighted_flow + 1e-10)
        return (blurred_flow - blurred_flow.min()) / (blurred_flow.max() - blurred_flow.min() + 1e-10)
    
    def _filter_video(self, video: list[np.ndarray], flow: list[np.ndarray]) -> list[np.ndarray]:
        return [self._filter_image(video[i], flow[i]) for i in range(len(video))]
    
    def _filter_stereo_video(self, video: tuple[list[np.ndarray], list[np.ndarray]], flow: tuple[list[np.ndarray], list[np.ndarray]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self._filter_video(video[0], flow[0]), self._filter_video(video[1], flow[1])
    
    def _filter_mono_video(self, video: list[np.ndarray], flow: list[np.ndarray]) -> list[np.ndarray]:
        return self._filter_video(video, flow)
    
    def __call__(
        self, 
        img: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | np.ndarray | tuple[np.ndarray, np.ndarray],
        flow: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | np.ndarray | tuple[np.ndarray, np.ndarray]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Filter a mono or stereo video using the local contrast"""

        if isinstance(img, tuple):
            if not isinstance(flow, tuple):
                raise ValueError("Flow must be a tuple if the image is a tuple")
            if isinstance(img[0], torch.Tensor):
                img_array = (img[0].permute(0, 2, 3, 1).detach().cpu().numpy(), img[1].permute(0, 2, 3, 1).detach().cpu().numpy())
                flow_array = (flow[0].permute(0, 2, 3, 1).detach().cpu().numpy(), flow[1].permute(0, 2, 3, 1).detach().cpu().numpy())
                transform_2_torch = True
            else:
                transform_2_torch = False
            filtered_flow = self._filter_stereo_video(img_array, flow_array)
            
            if transform_2_torch:
                filtered_flow = ([torch.from_numpy(frame).permute(2, 0, 1) for frame in filtered_flow[0]], [torch.from_numpy(frame).permute(2, 0, 1) for frame in filtered_flow[1]])
                filtered_flow = (torch.stack(filtered_flow[0]).to(flow[0].device), torch.stack(filtered_flow[1]).to(flow[1].device))
                
        else:
            if isinstance(img, torch.Tensor):
                img_array = img.permute(0, 2, 3, 1).detach().cpu().numpy()
                flow_array = flow.permute(0, 2, 3, 1).detach().cpu().numpy()
                transform_2_torch = True
            else:
                transform_2_torch = False

            filtered_flow = self._filter_mono_video(img_array, flow_array)
            
            if transform_2_torch:
                filtered_flow = [torch.from_numpy(frame).permute(2, 0, 1) for frame in filtered_flow]
                filtered_flow = torch.stack(filtered_flow).to(flow.device)

        return filtered_flow


