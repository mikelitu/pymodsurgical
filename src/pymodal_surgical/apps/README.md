# User applications

This section refers to the different user applications available in this library.

## Installation

To use the applications you will need some additional dependencies that are included in the [apps] dependencies. For this make sure to install them by calling:

```sh
pip install ./dist/pymodal_surgical.0.1.0.tar.gz[apps]
```

This will install [pygame](https://www.pygame.org/news), necessary for *Interactive Demo* application, and [PySide6](https://pypi.org/project/PySide6/), necessary for *Video Analysis*.

## Video Analysis


This applications is the main application to generate mode shapes and configuration files for the rest of the applications. Unlike the other applications it requires from *Qt6.6.2*, I recommend using the [qt installer](https://download.qt.io/official_releases/online_installers/). You will need to add the *QtMultimedia* extension to launch the video reproductor for the app. 

To launch the application, there are no flags for this user application.    

```sh
python -m pymodal_surgical.apps.video_analysis
```

You can check the source code at [video_analysis](video_analysis).

## Force Estimator

This application estimates the constraint force needed to translate the pixel grid between two different frames or a frame sequence. It requires from two variables to execute:

* **mode_shape_config**: This configuration contains the video path of the cyclic motion for the calculation of the mode shapes. If it already exists, the application will used the cached directory to calculate the forces. You can generate this config file using the *Video Analyser* application.

* **force_video_config**: This configuration contains the video to analyse, if it is not provided the applications will analyse the video from the mode shapes. We also define the directory to save the force grid, if the force is not simplified to a single weighted vector.

The command will be:

```sh
python -m pymodal_surgical.apps.force_estimator --mode-shape-config <mode-shape-config-dict> --force-video-config <force-video-config-dict>
```

You can check the source code at [force_estimator](force_estimator).

## Interactive Demo

This applications allows to create single image interactive demo, that can be controlled using either the mouse or a haptic device (*Touch X*). The simulation provides at the moment a single pixel direct manipulation and an *ODE Euler Solver* to handle the transition to resting state after a force is applied. The application contains the following flags:

* **demo_config**: This is a *.json* file that contains all the variables to initiate the interactive demo. There is no *ModeShapeCalculator* integration at the moment so it would not load the data from a cached directory. This file can be manually created or generated using with the *Video Analysis* application available for this package.

```sh
python -m pymodal_surgical.apps.interactive_demo --demo-config <path-to-config-file>
```

You can check the source code at [interactive_demo](interactive_demo).

