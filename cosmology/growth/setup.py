#!/usr/bin/env python

import os

from cosmology._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from cython_gsl import get_cython_include_dir, get_libraries, get_library_dir
    
    config = Configuration('growth', parent_package, top_path)

    cython(['core.pyx'], working_path=base_path)
    cython(['power.pyx'], working_path=base_path)
    cython(['correlation.pyx'], working_path=base_path)
    cython(['lensing.pyx'], working_path=base_path)

    config.add_extension('core', sources=['core.c', 'transfer.c', 'power_tools.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir(), ''],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])

    config.add_extension('power', sources=['power.c', 'transfer.c', 'power_tools.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir(), ''],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])
    
    config.add_extension('correlation', sources=['correlation.c', 'power_tools.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir(), ''],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])

    config.add_extension('lensing', sources=['lensing.c', 'power_tools.c', 'transfer.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir(), ''],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])

    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    
    setup(maintainer='cosmology Developers',
          author='cosmology Developers',
          description='growth',
          **(configuration(top_path='').todict())
          )