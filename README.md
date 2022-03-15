This repository contains the machine learning code used in the paper: Multi-source seamless lightning nowcasting with recurrent deep learning, to be submitted.

# Installation

You need NumPy, Scipy, Matplotlib, Tensorflow, Numba, Dask and NetCDF4 for Python.

Clone the repository, then, in the main directory, run
```bash
$ python setup.py develop
```
(if you plan to modify the code) or
```bash
$ python setup.py install
```
if you just want to run it.

# Downloading data

The dataset can be found at the following Zenodo repository: https://doi.org/10.5281/zenodo.6325370

Download the NetCDF file. You can place it in the `data` directory but elsewhere on the disk works too.

# Running

Go to the `scripts` directory and start an interactive shell. There, you can find `training.py` that contains the script you need for training and `plots_lightning.py` that produces the plots from the paper.
