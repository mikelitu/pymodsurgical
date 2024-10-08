PyModSurgical
=========================

This repository contains the source code for the pymodal_surgical package. This package is a Python implementation of the modal analysis method for soft tissue based on surgical videos. The implementation is based on previously published work `[1]`.

Installation
-------------
To install the package in your personal project, run the following command:

```shell
pip install git+https://github.com/mikelitu/surgical-video-modal-analysis.git
```

Building
-------------
To build the package, clone the repository and run the following commands in the root directory. I highly recommend using a designated *conda* environment as the package requires multiple dependencies. You can see a tutorial on how to setup an environment in [here](https://github.com/mikelitu/cheat-sheets/tree/main/Python-VSCode):

```shell
python -m build
pip install ./dist/pymodal_surgical-0.1.0.tar.gz
```

Applications
-------------
The package can be used for the following applications:

1. **Video Modal Analysis**: The package can be used to extract the modal basis of a soft tissue from a video. The modal basis can be used to simulate the deformation of the tissue or to calculate the forces from a givern deformation in a direct manipulation video.

2. **Manipulation analysis**: The package provides a tool to determine the camera projected force in an image based on the modal basis of the tissue and the optical flow of an image sequence. The application can used the previously cached modal basis or calculate it on the fly. The application receives a video and user selected ROI of pixels. It can either output a single force vector that represents the mean force on the area or a force vector for each pixel in the ROI.

3. **Interactive demo**: The package provides the possibility to generate realistic looking simulations of soft tissue deformation. The user can interact with the simulation by applying forces to the tissue and observing the deformation. The simulation is based on the modal basis of the tissue and the optical flow of the image sequence. The simulation uses a single frame and right now only provides the possibility of interacting with a single pixel.

4. **Video synthesis**: The package provides the possibility to synthesize videos given a command of an interaction. The user can determine the duration, intensity and direction of the contact. 

For more information refer to [documentation](src\pymodal_surgical\apps\README.md).


Examples
---------

The package provides a set of examples that demonstrate the usage of the package. The examples are located in the `examples` directory. The examples are:

* **Liver stereo video**: The example demonstrates the modal analysis of a stereo video of a liver. The example shows how to extract the modal basis of the liver from the video and how to simulate the deformation of the liver using the interactive demo application.

* **Beating heart video**: The example demonstrates the modal analysis of a video of a beating heart mono video. The example shows how to extract the modal basis of the heart from the video and how to simulate the deformation of the heart using the interactive demo application.

The data for the examples is not included in the repository, but can be downloaded from Google Drive and placed in the `data` directory. To download the data, run the following command in the root directory:

```shell
pip install gdown
gdown 'https://drive.google.com/uc?export=downloads&id=1UcIoMHDG3-5va0y7RWRAr54ZSIBuvYUF' -O data.zip
unzip data.zip -d data
rm data.zip
``` 

Testing
--------

This platform has been tested with custom videos and openly available real surgical videos from the Hamlyn Centre Laparoscopic / Endoscopic Video Dataset `[2]`. The videos are not included in the repository, but can be downloaded from the [official website]( https://hamlyn.doc.ic.ac.uk/vision/).

The package has been tested on **Ubuntu 20.04** and **Windows 10** with **Python 3.11**. The package has not been tested on other platforms or Python versions.


Docker container
------------------

Coming soon...


References
-----------

    [1] Davis, Abe, Justin G. Chen, and Frédo Durand. "Image-space modal bases for plausible manipulation 
    of objects in video." ACM Transactions on Graphics (TOG) 34.6 (2015): 1-7.

    [2] Peter Mountney, Danail Stoyanov and Guang-Zhong Yang: Three-Dimensional Tissue Deformation Recovery and Tracking: Introducing techniques based on laparoscopic or endoscopic images. IEEE Signal Processing Magazine. 2010 July. Volume: 27. Issue: 4. pp. 14-24.
    
