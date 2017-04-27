
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

import numpy as np

import unittest

from  extended_stencil import *

class TestDispersion2D(unittest.TestCase):
    def setUp(self):
        self.stencil = StencilDivFree2D()
        self.dispersion = Dispersion2D(1, 100, self.stencil)

    def test_omega(self):
        p = [-0.10,-0.60]
        dt = 0.3
        assert self.dispersion.stencil_ok([dt,*p]) > 0
        assert self.dispersion.dt_ok([dt,*p]) > 0
        omega = self.dispersion.omega([dt,*p])
        assert not np.any(np.isnan(omega))


class TestDispersion2DFree(unittest.TestCase):
    def setUp(self):
        self.stencil = StencilFree2D()
        self.dispersion = Dispersion2D(1, 100, self.stencil)

    def test_bad_free_stencil(self):
        p = [ 0.3, -0.3,0.3, 0.7]
        dt = 0.3
        assert self.dispersion.stencil_ok([dt,*p]) < 0


class TestDispersion3D(unittest.TestCase):
    def setUp(self):
        self.stencil = StencilDivFree3D()
        self.dispersion = Dispersion3D(1, 1, 100, self.stencil)

    def test_omega(self):
        p = [-0.10,-0.60,0.2]
        dt = 0.3
        assert self.dispersion.stencil_ok([dt,*p]) > 0
        assert self.dispersion.dt_ok([dt,*p]) > 0
        omega = self.dispersion.omega([dt,*p])
        assert not np.any(np.isnan(omega))

