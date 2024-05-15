import argparse
import json
from pathlib import Path
from .interactive_demo import InteractiveDemo

def main():
    argparser = argparse.ArgumentParser(description="Run the interactive demo.")
    argparser.add_argument("--config", type=str, default="liver", help="Name of the demo.")
    argparser.add_argument("--control", type=str, default="mouse", help="Control method for the demo. Options: 'mouse', 'haptic'")


    # Parse the command line arguments
    args = argparser.parse_args()

    # Retrieve the name and control method from the command line arguments
    config_file = args.config
    control = args.control

    # Load the configuration file
    config_file = Path(config_file)
    with open(config_file) as f:
        demo_config = json.load(f)

    # Create an instance of the InteractiveDemo class
    demo = InteractiveDemo(control_type=control, **demo_config)

    # Run the demo
    demo.run()

if __name__ == "__main__":
    main()