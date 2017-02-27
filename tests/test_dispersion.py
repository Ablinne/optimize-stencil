
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

