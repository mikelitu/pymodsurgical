=========================
Surgical Modal Analysis
=========================

This repository contains the source code for the pymodal_surgical package. This package is a Python implementation of the modal analysis method for soft tissue based on surgical videos. The implementation is based on previously published work :ref:`[1]`.

Installation
-------------
To install the package, clone the repository and run the following commands in the root directory. I highly recommend using a designated *conda* environment as the package requires multiple dependencies. You can see a tutorial on how to setup an environment in `here <https://github.com/mikelitu/cheat-sheets/tree/main/Python-VSCode>`_:

.. code:: sh

    python -m build
    pip install ./dist/pymodal_surgical-0.1.0.tar.gz


References
-----------
.. _[1]:

    [1] Davis, Abe, Justin G. Chen, and Frédo Durand. "Image-space modal bases for plausible manipulation 
    of objects in video." ACM Transactions on Graphics (TOG) 34.6 (2015): 1-7.
