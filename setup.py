from distutils.core import setup
import setuptools
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

name = 'usmltransform'
version = '0.1'
release = '0.1.1'

setup(
    name=name,
    author='Sebastian Semper',
    version=release,
    include_package_data=True,
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "usmltransform.CudaBlockTwoLevelToeplitz",
            sources=[
                "usmltransform/CudaBlockTwoLevelToeplitz.pyx",
            ],
            language="c++",
            include_dirs=[
                '/home/cuda/git/fastmat/fastmat',
                '/home/cuda/git/fastmat/fastmat/core',
                numpy.get_include()
            ],
            libraries=[],
            extra_link_args=[
                '-lcuda',
                '-lcudart',
                '-lcufft',
                'usmltransform/BlockTwoLevelKernel.o'
            ],
        )
    ]
)
