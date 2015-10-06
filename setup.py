#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension
from Cython.Build import cythonize

import numpy


source_files = ["meanet/meanet.pyx"]
include_dirs = [numpy.get_include()]
extensions = [Extension("meanet.meanet",
                        sources=source_files,
                        include_dirs=include_dirs)]
                        
with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = ['cython', 'numpy']


setup(
    name='meanet',
    version='0.1.0',
    description="Utilities for converting MEA data into networks",
    long_description=readme + '\n\n' + history,
    author="Larry Eisenman",
    author_email='leisenman@wustl.edu',
    url='https://github.com/lneisenman/meanet',
    packages=[
        'meanet',
    ],
    package_dir={'meanet':
                 'meanet'},
    package_data={'': ['*.pyx', '*.pxd', '*.h', '*.txt', '*.dat', '*.csv']},
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='meanet',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Cython',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    ext_modules=cythonize(extensions),
)
