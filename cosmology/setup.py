#!/usr/bin/env python

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('cosmology', parent_package, top_path)

    config.add_subpackage('utils')
    config.add_subpackage('parameters')
    config.add_subpackage('evolution')
    config.add_subpackage('growth')
    config.add_subpackage('halo')

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)