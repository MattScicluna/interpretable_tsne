"""
python setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["src/interpretable_tsne/_utils.pyx",
                           "src/interpretable_tsne/_quad_tree.pyx",
                           "src/interpretable_tsne/_bintree.pyx",
                           "src/interpretable_tsne/_barnes_hut_tsne.pyx",
                           "src/interpretable_tsne/_grad_comps.pyx"]),
    include_dirs=[numpy.get_include()]
)
