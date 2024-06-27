from pymodal_surgical.apps.video_generator.video_generator import VideoGenerator
import argparse
import json
import sys


def main():
    argparser = argparse.ArgumentParser(description="Generate a video of a modal interaction.")
    argparser.add_argument("--config", type=str, required=True, help="Configuration file for the video generator")

    # Parse the command line arguments
    args = argparser.parse_args()

    # Retrieve the configuration file from the command line arguments
    config_file = args.config

    try:
        with open(config_file) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found. Please provide a valid configuration file.")
    
    # Create an instance of the VideoGenerator class
    generator = VideoGenerator(config)
    generator.generate_video()


if __name__ == "__main__":
    main()
