[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2562741.svg)](https://doi.org/10.5281/zenodo.2562741)

# Ultrasonic Multilevel Transforms in CUDA

## Features
 - Implementation of the 3D pulse echo model as a linear transform on CUDA.
 - Fourier Subsampling class of the A-scans using above CUDA implementation as a forward model
 - This allows matrix-free sparse recovery for full 3D volumetric data

## Getting things to work
To compile everything run `build.sh` after supplying the correct path to your [fastmat](https://github.com/EMS-TU-Ilmenau/fastmat) directory in the `setup.py` file. Of course you need Cuda, fastmat and Cython installed in your local Python path as well and provide all necessary include and library paths.

After running `./build.sh` you can invoke `pip install -e .` and import it via `import usmltransform`

See the `test.py` file for further clarification.
