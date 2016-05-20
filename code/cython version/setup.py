from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Dual Decomposition: In Place Version',
  ext_modules = cythonize(["subproblemInPlace.pyx"]),
)