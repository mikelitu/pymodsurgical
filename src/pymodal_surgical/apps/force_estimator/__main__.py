from .force_estimator import ForceEstimator
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--force_video_path", type=str, required=True)
argparser.add_argument("--mode_shape_video_path", type=str, required=True)
argparser.add_argument("--force_video_config", type=str, required=True)
argparser.add_argument("--mode_shape_config", type=str, required=True)

args = argparser.parse_args()

def main():
    estimator = ForceEstimator(
        force_estimation_video_path=args.force_video_path,
        force_estimation_config=args.force_video_config,
        mode_shape_video_path=args.mode_shape_video_path,
        mode_shape_config=args.mode_shape_config
    )



if __name__ == "__main__":
    main()