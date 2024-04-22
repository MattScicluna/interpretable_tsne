"""
python setup.py build_ext --inplace
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


# Assuming .pyx files are in 'src/interpretable_tsne'
extensions = [
        Extension("interpretable_tsne._utils", 
                  ["src/interpretable_tsne/_utils.pyx"], 
                  include_dirs=[numpy.get_include()]),
        Extension("interpretable_tsne._quad_tree", 
                  ["src/interpretable_tsne/_quad_tree.pyx"], 
                  include_dirs=[numpy.get_include()]),
        Extension("interpretable_tsne._bintree", 
                  ["src/interpretable_tsne/_bintree.pyx"], 
                  include_dirs=[numpy.get_include()]),
        Extension("interpretable_tsne._barnes_hut_tsne", 
                  ["src/interpretable_tsne/_barnes_hut_tsne.pyx"], 
                  include_dirs=[numpy.get_include()]),
        Extension("interpretable_tsne._grad_comps", 
                  ["src/interpretable_tsne/_grad_comps.pyx"], 
                  include_dirs=[numpy.get_include()])
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include(), 'src/interpretable_tsne/']
)