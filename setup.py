#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Fuzhan Rahmanian",
    author_email='fuzhanrahmanian@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Package for estimation of the state of health prediction of battery materials",
    entry_points={
        'console_scripts': [
            'arcana=arcana.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='arcana',
    name='arcana',
    packages=find_packages(include=['arcana', 'arcana.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url="https://github.com/basf/ARCANA",
    version='1.0.0',
    zip_safe=False,
)
