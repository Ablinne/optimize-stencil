#!/usr/bin/env python3


from setuptools import setup

setup(name='OptimizeStencil',
      version='0.1',
      description='OptimizeStencil',
      author='Alexander Blinne, David Schinkel',
      author_email='alexander@blinne.net',
      url='https://git.tpi.uni-jena.de/albn/optimize_stencil',
      packages=['extended_stencil'],
      scripts=['optimize_stencil.py', 'calculate_omega.py'],
      install_requires=['numpy', 'scipy', 'psutil']
     )
