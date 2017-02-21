
import numpy as np

import unittest

from  extended_stencil import Dispersion2D, StencilFixed2D

class TestStencilFixed2D(unittest.TestCase):
    def setUp(self):
        self.stencil = StencilFixed2D()

    def test_args(self):
        p = [0.1,-0.10,-0.60]
        assert self.stencil.args_ok(p) > 0

    def test_args_bad(self):
        p = [0.1,0.10,0.60]
        assert self.stencil.args_ok(p) < 0
