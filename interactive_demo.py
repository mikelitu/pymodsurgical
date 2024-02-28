import pygame
from motion_spectrum import calculate_motion_spectrum, resize_spectrum_2_reference
from video_reader import VideoReader, RetType
from displacement import calculate_deformation_map
import torch
from complex import motion_spectrum_2_complex
import json
from pathlib import Path, PosixPath
from masking import Masking
import math
from torchvision.utils import flow_to_image
from optical_flow import warp_flow

rad = math.pi / 180

def get_motion_spectrum_from_video(
    video_path: PosixPath | str, 
    K: int = 16,
    start: int = 0,
    end: int = 0,
    filtering: bool = True, 
    masking: bool = True
) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], VideoReader]:
    
    video_path = Path(video_path)
    with open("videos/metadata.json", "r") as f:
        video_config = json.load(f)
    
    video_reader = VideoReader(video_path, video_config, return_type=RetType.NUMPY)
    frames = video_reader.read(start, end)
    
    if masking:
        mask = Masking(video_reader.video_config[video_reader.video_path.stem]["mask"], video_reader.video_type)
    else:
        mask = None

    if isinstance(frames, tuple):
        motion_spectrum = (calculate_motion_spectrum(frames[0], K, filtered=filtering, mask=mask, camera_pos="left", save_flow_video=True), calculate_motion_spectrum(frames[1], K, filtered=filtering, mask=mask, camera_pos="right"))
    else:
        motion_spectrum = calculate_motion_spectrum(frames, K, filtered=filtering, mask=mask)
    
    return motion_spectrum, video_reader

def arrow(screen, lcolor, tricolor, start, end, trirad, thickness=10):
    pygame.draw.line(screen, lcolor, start, end, thickness)
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi/2
    pygame.draw.polygon(screen, tricolor, ((end[0] + trirad * math.sin(rotation),
                                        end[1] + trirad * math.cos(rotation)),
                                       (end[0] + trirad * math.sin(rotation - 120*rad),
                                        end[1] + trirad * math.cos(rotation - 120*rad)),
                                       (end[0] + trirad * math.sin(rotation + 120*rad),
                                        end[1] + trirad * math.cos(rotation + 120*rad))))

def main():
    video_path = "videos/liver_stereo.avi"
    K = 100
    motion_spectrum, video_reader = get_motion_spectrum_from_video(video_path, K)
    reference_frame = video_reader.read_frame(0)

    pygame.init()
    screen = pygame.display.set_mode((reference_frame[0].shape[0], reference_frame[0].shape[1]))
    motion_spectrum = resize_spectrum_2_reference(motion_spectrum, reference_frame[0])
    motion_spectrum = motion_spectrum_2_complex(motion_spectrum)

    pygame.display.set_caption("Deformation Map")
    image = pygame.surfarray.make_surface(reference_frame[0])
    screen.blit(image, (0, 0))
    pygame.display.flip()
    running = True
    on_click = False
    pixel = None
    cur_pos = None
    clock = pygame.time.Clock()
    
    while running:
        clock.tick(30)
        screen.fill((1, 1, 1))
        
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                on_click = True
                pixel = event.pos
                cur_pos = event.pos
                displacement = torch.tensor([0, 0])
            if event.type == pygame.MOUSEBUTTONUP:
                on_click = False
            if event.type == pygame.MOUSEMOTION and on_click:
                displacement = torch.tensor([event.pos[0] - pixel[0], event.pos[1] - pixel[1]])
                displacement = displacement.float() / displacement.float().norm(2)
                cur_pos = event.pos
            if event.type == pygame.QUIT:
                running = False
        
        
        if cur_pos is not None and pixel is not None and on_click:
            deformation_map, _ = calculate_deformation_map(motion_spectrum[0], displacement, pixel)
            deformed_frame = warp_flow(torch.from_numpy(reference_frame[0]).permute(2, 0, 1).float().cuda(), deformation_map.cuda())
            deformed_frame_surf = pygame.surfarray.make_surface(deformed_frame)
            screen.blit(deformed_frame_surf, (0, 0))
            arrow(screen, (0, 0, 0), (0, 0, 0), pixel, cur_pos, 10)
        else:
            screen.blit(image, (0, 0))

        


        pygame.display.flip()  
    pygame.quit()

if __name__ == "__main__":
    main()

