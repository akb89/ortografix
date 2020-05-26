#!/usr/bin/env python3
"""ortografix setup.py.

This file details modalities for packaging the ortografix application.
"""

from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='ortografix',
    description='Seq2seq model with attention for automatic orthographic simplification',
    author=' Alexandre Kabbach',
    author_email='akb@3azouz.net',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.7.0',
    url='https://github.com/akb89/ortografix',
    download_url='https://github.com/akb89/ortografix',
    license='MIT',
    keywords=['seq2seq', 'ortographic simplification'],
    platforms=['any'],
    packages=['ortografix', 'ortografix.logging', 'ortografix.exceptions',
              'ortografix.utils', 'ortografix.model'],
    package_data={'ortografix': ['logging/*.yml', 'resources/*']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ortografix = ortografix.main:main'
        ],
    },
    install_requires=['pyyaml>=4.2b1', 'torch==1.5.0', 'textdistance==4.2.0'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    zip_safe=False,
)
