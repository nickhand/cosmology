from setuptools import setup, find_packages
import os

setup(
    name='cosmology',
    version='1.0',
    author='Nick Hand',
    author_email='nicholas.adam.hand@gmail.com',
    packages=find_packages(),
    description='python package for cosmological calculations',
    long_description=open('README.md').read()
)