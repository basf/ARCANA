#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup(
    author="Fuzhan Rahmanian",
    author_email='fuzhanrahmanian@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: GPU :: NVIDIA CUDA :: 11.0',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description="Package for the estimation of early stage battery degradation mechanism and lifecycle monitoring",
    entry_points={
    },
    install_requires=requirements,
    license="MIT",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='arcana',
    name='arcana-batt',
    packages=find_packages(include=['arcana', 'arcana.*']),
    package_data={'arcana\plots\styles': ['*.mplstyle']},
    url="https://github.com/basf/ARCANA",
    version='1.0.0',
    zip_safe=False,
)
