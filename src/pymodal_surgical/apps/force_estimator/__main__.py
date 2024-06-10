from .force_estimator import ForceEstimator
import argparse
from pathlib import Path
import json
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, required=True)
argparser.add_argument("--plot", action="store_true", help="Plot the force data")

args = argparser.parse_args()


def open_config(config_path: str | Path) -> dict:

    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def check_mode(config: dict):
    force_mode = config["force_estimation_config"]["mode"]
    if force_mode not in ["simple", "texture"]:
        raise ValueError(f"Invalid mode {force_mode}. Options are 'simple' and 'texture'")
    
    return force_mode


def main():

    if not args.config:
        raise ValueError("Config file not provided")
    


    config = open_config(args.config)

    force_mode = check_mode(config)

    estimator = ForceEstimator(
        **config
    )

    if args.idx_2 == 0:
        args.idx_2 = len(estimator.force_video_reader)

    forces = np.zeros((args.idx_2 - args.idx_1, 2))

    for idx in range(args.idx_1, args.idx_2):
        force = estimator.calculate_force(idx, idx + 1)
        forces[idx - args.idx_1] = force
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    force.save(output_dir / f"force_{args.idx_1}_{args.idx_2}.npy")

    if args.plot:
        if force_mode != "simple":
            raise ValueError("Cannot plot force data for texture mode")
        
        forces = (forces - forces.mean(axis=0)) / (forces.std(axis=0) + 1e-6)
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        time = (1/30) * np.arange(0, forces.shape[0], 1)
        for i in range(2):
            axs[i].plot(time, forces[:, i])
        
        fig.supxlabel("Time (s)")
        fig.supylabel("Normalized Force")
        plt.show()

if __name__ == "__main__":
    main()