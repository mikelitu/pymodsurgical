from interactive_demo import InteractiveDemo
import json
import argparse
from pathlib import Path

argparser = argparse.ArgumentParser(description="Run the interactive demo.")
argparser.add_argument("--name", type=str, default="liver", help="Name of the demo.")
argparser.add_argument("--control", type=str, default="mouse", help="Control method for the demo. Options: 'mouse', 'haptic'")



def main():
    # Parse the command line arguments
    args = argparser.parse_args()

    # Retrieve the name and control method from the command line arguments
    name = args.name
    control = args.control

    # Load the configuration file
    config_file = Path('config/demos.json')
    with open(config_file) as f:
        config = json.load(f)

    # Create an instance of the InteractiveDemo class
    demo = InteractiveDemo(control_type=control, **config[name])

    # Run the demo
    demo.run()


if __name__ == "__main__":
    main()