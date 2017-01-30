from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl
import numpy as np

extensions = [
    Extension("expansion_hard_disks.simulation",
              sources=["expansion_hard_disks/simulation.pyx"],
              language="c", libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir(), np.get_include()])
]

setup(
    name='expansion_hard_disks',
    version='1.0',
    url='',
    license='',
    author='Andrea Giometto',
    author_email='giometz@gmail.com',
    description='Range expansion of cells modeled as hard disks.',
    include_dirs = [cython_gsl.get_include(), np.get_include()],
    ext_modules = cythonize(extensions, annotate=True)
)