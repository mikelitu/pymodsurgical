from .force_estimator import ForceEstimator
import argparse
from pathlib import Path
import json
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, required=True)
argparser.add_argument("--force_video_config", type=str, required=True)
argparser.add_argument("--mode_shape_config", type=str, required=True)
argparser.add_argument("--output_dir", type=str, required=True)
argparser.add_argument("--idx_1", type=int, help="Index of the first frame", default=0)
argparser.add_argument("--idx_2", type=int, help="Index of the second frame", default=0)
argparser.add_argument("--plot", action="store_true", help="Flag to plot the force data", default=False)

args = argparser.parse_args()


def open_config(config_path: str | Path) -> dict:

    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def main():

    config = open_config(args.config)

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
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        time = (1/30) * np.arange(0, forces.shape[0], 1)
        for i in range(2):
            axs[i].plot(time, forces[:, i])
        
        fig.supxlabel("Time (s)")
        fig.supylabel("Force")
        plt.show()

if __name__ == "__main__":
    main()