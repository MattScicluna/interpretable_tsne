"""
python setup.py build_ext --inplace
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

#setup(
#    ext_modules=cythonize(["src/interpretable_tsne/_utils.pyx",
#                           "src/interpretable_tsne/_quad_tree.pyx",
#                           "src/interpretable_tsne/_bintree.pyx",
#                           "src/interpretable_tsne/_barnes_hut_tsne.pyx",
#                           "src/interpretable_tsne/_grad_comps.pyx"]),
#    include_dirs=[numpy.get_include()],
#    compiler_directives={'language_level' : "3"}
#)

setup(
    ext_modules=cythonize([
        Extension("interpretable_tsne._utils", 
                  ["src/interpretable_tsne/_utils.c"], 
                  include_dirs=[numpy.get_include()]),
        Extension("interpretable_tsne._quad_tree", 
                  ["src/interpretable_tsne/_quad_tree.c"], 
                  include_dirs=[numpy.get_include()]),
        Extension("interpretable_tsne._bintree", 
                  ["src/interpretable_tsne/_bintree.c"], 
                  include_dirs=[numpy.get_include()]),
        Extension("interpretable_tsne._barnes_hut_tsne", 
                  ["src/interpretable_tsne/_barnes_hut_tsne.c"], 
                  include_dirs=[numpy.get_include()]),
        Extension("interpretable_tsne._grad_comps", 
                  ["src/interpretable_tsne/_grad_comps.c"], 
                  include_dirs=[numpy.get_include()])
    ]),
    compiler_directives={'language_level' : "3"}
)
