import numpy
from distutils.core import setup, Extension

module = Extension('evcreader',
                   sources=['evcreader.c'],
                   include_dirs=[numpy.get_include()])

setup(name='EVCReader',
      version='1.0',
      description='This is a package for fast reading .evc files',
      ext_modules=[module])
