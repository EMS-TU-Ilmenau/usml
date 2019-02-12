# Make things fast! Ultrasonic things in CUDA

## Features

 - Implementation of the 3D forward model as a linear transform on CUDA.
 - Fourier Subsampling class of the a-scans using above CUDA implementation as a forward model


## Getting things to work
To compile everything run `build.sh` after supplying the correct path to your fastmat git clone directory in the `setup.py` file. Of course you need Cuda, fastmat and Cython installed as well and provide all necessary include and library paths.

After running `./build.sh` you can invoke `pip install -e .` and import it via `import usmltransform`

See the `test.py` file for further clarification.
