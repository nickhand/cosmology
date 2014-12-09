#!/usr/bin/env python
import os

from cosmology._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from cython_gsl import get_include, get_libraries, get_library_dir
    
    config = Configuration('halo', parent_package, top_path)

    cython(['bias.pyx'], working_path=base_path)
    
    config.add_extension('bias', sources=['bias.c', 'halo_tools.c'],
                         include_dirs=[get_numpy_include_dirs(), get_include(), ''],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])
                         
    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    
    setup(maintainer='cosmology Developers',
          author='cosmology Developers',
          description='halo',
          **(configuration(top_path='').todict())
          )