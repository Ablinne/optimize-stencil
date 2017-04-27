#!/usr/bin/env python3

# This file is part of the optimize_stencil project
#
# Copyright (c) 2017 Alexander Blinne
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup

setup(name='OptimizeStencil',
      version='0.1',
      description='Optimize stencils for extended Maxwell Solvers',
      author='Alexander Blinne, David Schinkel',
      author_email='alexander@blinne.net',
      url='https://git.tpi.uni-jena.de/albn/optimize_stencil',
      packages=['extended_stencil'],
      scripts=['optimize_stencil.py', 'calculate_omega.py'],
      install_requires=['numpy', 'scipy'],
      extras_require= { 'CountPhysicalCores': ['psutil'] }
     )
