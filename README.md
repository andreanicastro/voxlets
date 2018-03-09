# Structured Prediction of Unobserved Voxels from a Single Depth Image

    @inproceedings{firman-cvpr-2016,
      author = {Michael Firman and Oisin Mac Aodha and Simon Julier and Gabriel J Brostow},
      title = {{Structured Completion of Unobserved Voxels from a Single Depth Image}},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year = {2016}
    }

## Downloading the dataset

The tabletop dataset can be downloaded from:

https://www.dropbox.com/s/bqhub295475najw/voxlets_dataset.zip?dl=0

This is a 395MB zip file. You will have to change some of the paths in `src/pipeline/real_data_paths.py` to the location you have extracted the dataset to.

The voxel regions used to evaluate each of the scenes can be downloaded from here:

https://www.dropbox.com/s/x8xc27jqbvqoal9/evaluation_regions.zip

### Getting started with the tabletop dataset

An example iPython notebook file loading a ground truth TSDF grid and plotting on the same axes as a depth image is given in `src/examples/Voxel_data_io_example.ipynb`

## Code overview

The code is roughly divided into three areas:

1. `src/common/` is a Python module containing all the classes and functions used for manipulation of the data and running of the routines. Files included are:

    - `images.py` - classes for RGBD images and videos
    - `camera.py` - a camera class, enabling points in 3D to be projected into a virtual depth image and vice versa
    - `mesh.py` - a class for 3D mesh data, including IO and marching cubes conversion
    - `voxel_data.py` - classes for 3D voxel grids, including various manipulation routines and ability to copy data between grids at different locations in world space
    - `carving.py` - classes for reconstructing 3D volumes from extrinsicly calibrated depth images.
    - `features.py` - classes for computing normals and other features from depth images
    - `random_forest_structured.py` - structured RF class
    - `scene.py` - class which contains a voxel grid and one or more images with corresponding coordinate frames.
    - `voxlets.py` - class for storing and predicting voxlets, and for doing the final reconstruction of the output voxel grid.

2. `src/pipeline/` - Contains scripts for loading data, performing processing and saving files out.

3. `src/examples/` - iPython notebooks containing examples of use of the data and code.

### Prerequisites

I have run this code using **Python 2** on a fairly up-to-date version of Anaconda on Ubuntu 14.04.
If you have a newer version of scipy, you will need to install weave seprately, which has been moved to its own project.


### How to run the pipeline on the tabletop dataset

Navigate to `src/pipeline`

    >> python 06_compute_pca.py
    >> python 08_extract_all_voxlets.py
    >> python 09_train_forest.py
    >> python 10_predict.py
    >> python 11_render.py


### How to run the pipeline on the synthetic NYU dataset

Navigate to `src/pipeline`

    >> python 06_compute_pca.py training_params_nyu_silberman.yaml
    >> python 08_extract_all_voxlets.py training_params_nyu_silberman.yaml
    >> python 09_train_forest.py training_nyu_silberman.yaml
    >> python 10_predict.py testing_params_nyu_silberman.yaml
    >> python 11_render.py testing_params_nyu_silberman.yaml
