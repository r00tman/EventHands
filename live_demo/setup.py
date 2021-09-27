#!/usr/bin/env python3
import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('live_cython.pyx'), include_dirs=[numpy.get_include()])
