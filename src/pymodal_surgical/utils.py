from pathlib import Path, PosixPath
import numpy as np
import torch

def create_save_dir(
    save_dir: PosixPath | str, 
    filename: str,
) -> PosixPath:
    if not isinstance(save_dir, PosixPath):
        save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    print(f"Saving to: {save_dir / filename}")
    return save_dir / filename
