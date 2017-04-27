
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

from types import SimpleNamespace
import unittest
from nose.tools import raises

from  extended_stencil import *


class TestGetStencil3D(unittest.TestCase):
    def test_Free(self):
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 0
        args.div_free = False
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilFree3D)


    def test_DivFree(self):
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 0
        args.div_free = True
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilDivFree3D)

        args.symmetric_beta = True
        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilDivFree3D)


    def test_Symmetric(self):
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 0
        args.div_free = False
        args.symmetric_beta = True

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilSymmetric3D)


    @raises(NotImplementedError)
    def test_CylinderDivFreeSymmetric(self):
        ### This is a contradiction as Cylinder + DivFree -> betazx != betaxz
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 1
        args.div_free = True
        args.symmetric_beta = True

        s = Stencil.get_stencil(args)


    def test_Cylinder(self):
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 1
        args.div_free = False
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilCylinder3D)


    @raises(NotImplementedError)
    def test_CylinderSymmetric(self):
        ### This is a contradiction as Cylinder + DivFree -> betazx != betaxz
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 1
        args.div_free = False
        args.symmetric_beta = True

        s = Stencil.get_stencil(args)


    def test_CylinderDivFree(self):
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 1
        args.div_free = True
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilCylinderDivFree3D)


    def test_Isotropic(self):
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 2
        args.div_free = False
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)

        assert isinstance(s, StencilIsotropic3D)


    def test_IsotropicDivFree(self):
        args = SimpleNamespace()
        args.dim = 3
        args.symmetric_axes = 2
        args.div_free = True
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilIsotropicDivFree3D)

        args.symmetric_beta = True
        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilIsotropicDivFree3D)



class TestGetStencil2D(unittest.TestCase):
    def test_Free(self):
        args = SimpleNamespace()
        args.dim = 2
        args.symmetric_axes = 0
        args.div_free = False
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilFree2D)

    def test_DivFree(self):
        args = SimpleNamespace()
        args.dim = 2
        args.symmetric_axes = 0
        args.div_free = True
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilDivFree2D)

    def test_Isotropic(self):
        args = SimpleNamespace()
        args.dim = 2
        args.symmetric_axes = 1
        args.div_free = False
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilIsotropic2D)

        args.symmetric_beta = True
        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilIsotropic2D)


    def test_Symmetric(self):
        args = SimpleNamespace()
        args.dim = 2
        args.symmetric_axes = 0
        args.div_free = False
        args.symmetric_beta = True

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilSymmetric2D)


    def test_Isotropic_DivFree(self):
        args = SimpleNamespace()
        args.dim = 2
        args.symmetric_axes = 1
        args.div_free = True
        args.symmetric_beta = False

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilIsotropicDivFree2D)

        args.symmetric_beta = True
        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilIsotropicDivFree2D)


    def test_Symmetric_DivFree(self):
        args = SimpleNamespace()
        args.dim = 2
        args.symmetric_axes = 0
        args.div_free = True
        args.symmetric_beta = True

        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilIsotropicDivFree2D)

        args.symmetric_axes = 1
        s = Stencil.get_stencil(args)
        assert isinstance(s, StencilIsotropicDivFree2D)
