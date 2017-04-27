
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

from collections import namedtuple
from enum import Enum
from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
from numpy.version import version as npver


npvmajor, npvminor, npvbugfix = [int(x) for x in npver.split('.')]

class namedarray:
    """Class representing a set of field names such that any list or array of numbers can be casted into a :class:`numpy.recarray`"""

    def __init__(self, field_names):
        """:param field_names: List of field names
        :type field_names: list"""
        if isinstance(field_names, str):
            self._fields = [field_name.strip() for field_name in field_names.split(',')]
        else:
            self._fields = field_names[:]
        self._nfields = len(self._fields)
        self.dtype = np.dtype([(field_name, float) for field_name in self._fields])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._fields)

    def __call__(self, args):
        """Convert a list or an array into  a :class:`numpy.recarray`

        :param args: List or Array to be converted
        :type args: iterable"""

        #print(self._fields, args)
        return np.asfarray(args).view(dtype = self.dtype).view(np.recarray)

    def convert(self, other):
        """Convert a :class:`numpy.recarray` to a :class:`numpy.recarray` with additional fields, filling the additional fields with 0.

        :param other: The :class:`numpy.recarray` to be converted.
        :type other: :class:`numpy.recarray`"""

        a = self(np.zeros(self._nfields))
        if npvminor >= 13:
            a[other.dtype.names] = other
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                a[:] = other
        return a

class StencilFlags(Enum):
    """An enumeration representing various flags a Stencil can carry."""

    DIM2 = 0
    """A stencil in two dimensions."""

    DIM3 = 1
    """A stencil in three dimensions."""

    DIVFREE = 2
    """A stencil that preserves :math:`\\operatorname{div}\\vec{B}=0`."""

    ISOTROPIC = 3
    """A stencil that is the same on all axis."""

    CYLINDERSYM = 4
    """A stencil that is the same on two out of three axes."""

    SYMMETRICBETA = 5
    """A stencil with symmetric :math:`\\beta_{ij}`."""

class Stencil(metaclass = ABCMeta):
    """Parent class representing all Stencils"""

    Parameters = None
    """:class:`namedarray` object representing the free parameters of the stencil"""

    Coefficients = None
    """:class:`namedarray` object representing the actual coefficients of the stencil"""

    stencils = {}
    """:class:`dict` that stores all :class:`Stencil` subclasses with its :class:`set` of flags"""

    def __init__(self):
        self._parameters = self.Coefficients._fields[:]
        for k in self._fixed_coefficients():
            try:
                self._parameters.remove(k)
            except ValueError:
                pass
            #print(k, self._parameters)
        self.Parameters = namedarray(self._parameters)

    @property
    def dof(self):
        """Read-only property giving the number of the degrees of freedom."""
        return len(self._parameters)

    def coefficients(self, args):
        """Method that converts the given parameters into the stencil coefficients.

        :param args: iterable of parameters
        :type args: iterable
        """
        c = self.Coefficients.convert(self.Parameters(args))
        self._fill_coefficients(c)
        return c

    def _fixed_coefficients(self):
        #print('_fixed_coeff')
        return []

    def _fill_coefficients(self, coefficients):
        pass

    @classmethod
    def register_subclasses(cls, sub = None):
        """This method will walk the tree of subclasses of :class:`Stencil` and fill :attr:`Stencil.stencils`.
        This method runs recursively.

        :param sub: Start from thus subclass
        :type sub: :class:`Stencil`"""
        if sub is None:
            sub = cls
        for subclass in sub.__subclasses__():
            cls.register_subclasses(subclass)
            cls.stencils[subclass.Flags] = subclass

    @classmethod
    def get_stencil(cls, args):
        """This method will do two things:

        1. Interpret the given args in order to determine the set of flags that the user wants
        2. Find and instantiate a :class:`Stencil` subclass that has the requested flags

        :param args: Namespace object containing all the arguments.
        :type args: namespace
        :return: Object of a subclass of :class:`Stencil`
        :rtype: :class:`Stencil`"""
        if cls.stencils == {}:
            cls.register_subclasses()

        requested_flags = set()
        if args.dim == 2:
            requested_flags.add(StencilFlags.DIM2)
        elif args.dim == 3:
            requested_flags.add(StencilFlags.DIM3)
        else:
            raise NotImplementedError

        if args.div_free:
            requested_flags.add(StencilFlags.DIVFREE)

        if args.symmetric_beta:
            requested_flags.add(StencilFlags.SYMMETRICBETA)

        if args.symmetric_axes == args.dim - 1:
            requested_flags.add(StencilFlags.ISOTROPIC)
        if args.symmetric_axes == 1 and args.dim == 3:
            requested_flags.add(StencilFlags.CYLINDERSYM)

        #print('requested_flags', requested_flags)
        #print('requested_flags', list(cls.stencils.keys()))

        flagsets = [flagset for flagset in cls.stencils.keys() if flagset >= requested_flags]
        if len(flagsets) == 0:
            raise NotImplementedError

        if len(flagsets) > 1:
            flagsets = sorted(flagsets, key = lambda x: len(x))

        #print('flagsets', flagsets)

        flags = flagsets[0]
        if flags > requested_flags:
            print('Chosen stencil also has flags: ', flags - requested_flags)
        return Stencil.stencils[flags]()


def get_stencil(args):
    """Module level function that will call :func:`Stencil.get_stencil`."""
    return Stencil.get_stencil(args)

class StencilFree2D(Stencil):
    """Most general two-dimensional stencil"""

    Coefficients = namedarray("dt, alphax, alphay, betaxy, betayx, deltax, deltay")
    Flags = frozenset([StencilFlags.DIM2])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['alphax', 'alphay']

    def _fill_coefficients(self, c):
        c.alphax = 1.0 - 2.0 * c.betaxy - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 3.0 * c.deltay
        super()._fill_coefficients(c)


class StencilDivFree2D(StencilFree2D):
    """Two dimensional stencil that preserves :math:`\\operatorname{div}\\vec{B}=0`."""

    Flags = StencilFree2D.Flags | frozenset([StencilFlags.DIVFREE])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betaxy', 'betayx']

    def _fill_coefficients(self, c):
        c.betaxy = c.deltay
        c.betayx = c.deltax
        super()._fill_coefficients(c)


class StencilSymmetric2D(StencilFree2D):
    """Two dimensional stencil with symmetric :math:`\\beta_{ij}`."""

    Flags = StencilFree2D.Flags | frozenset([StencilFlags.SYMMETRICBETA])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betayx']

    def _fill_coefficients(self, c):
        c.betayx = c.betaxy
        super()._fill_coefficients(c)


class StencilIsotropic2D(StencilSymmetric2D):
    """A two dimensional stencil that is the same on all axis."""

    Flags = StencilSymmetric2D.Flags | frozenset([StencilFlags.ISOTROPIC])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['deltay']

    def _fill_coefficients(self, c):
        c.deltay = c.deltax
        super()._fill_coefficients(c)


class StencilIsotropicDivFree2D(StencilIsotropic2D, StencilDivFree2D):
    """A two dimensional stencil that is the same on each axis and preserves :math:`\\operatorname{div}\\vec{B}=0`."""
    ### Note that the order of parent classes matters!
    ### MRO will call the inherited methods from left to right,
    ### setting delta_y = delta_x (Isotropic2D) first and than
    ### calculating the betas from the deltas (DivFree2D)

    Flags = StencilDivFree2D.Flags | StencilIsotropic2D.Flags



class StencilFree3D(Stencil):
    """Most general three-dimensional stencil"""

    Coefficients = namedarray("dt, alphax, alphay, alphaz, betaxy, betaxz, betayx, betayz, betazx, betazy, deltax, deltay, deltaz")
    Flags = frozenset([StencilFlags.DIM3])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['alphax', 'alphay', 'alphaz']

    def _fill_coefficients(self, c):
        c.alphax = 1.0 - 2.0 * c.betaxy - 2.0 * c.betaxz - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 2.0 * c.betayz - 3.0 * c.deltay
        c.alphaz = 1.0 - 2.0 * c.betazx - 2.0 * c.betazy - 3.0 * c.deltaz
        super()._fill_coefficients(c)


class StencilDivFree3D(StencilFree3D):
    """Three dimensional stencil that preserves :math:`\\operatorname{div}\\vec{B}=0`."""

    Flags = StencilFree3D.Flags | frozenset([StencilFlags.DIVFREE])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betaxy', 'betaxz', 'betayx', 'betayz', 'betazx', 'betazy']

    def _fill_coefficients(self, c):
        c.betaxy = c.deltay
        c.betaxz = c.deltaz
        c.betayx = c.deltax
        c.betayz = c.deltaz
        c.betazx = c.deltax
        c.betazy = c.deltay
        super()._fill_coefficients(c)


class StencilSymmetric3D(StencilFree3D):
    """Three dimensional stencil with symmetric :math:`\\beta_{ij}`."""

    Flags = StencilFree3D.Flags | frozenset([StencilFlags.SYMMETRICBETA])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betayx', 'betazx', 'betazy']

    def _fill_coefficients(self, c):
        c.betayx = c.betaxy
        c.betazx = c.betaxz
        c.betazy = c.betayz
        super()._fill_coefficients(c)


class StencilCylinder3D(StencilFree3D):
    """Three dimensional stencil that is the same on two out of three axes."""

    Flags = StencilFree3D.Flags | frozenset([StencilFlags.CYLINDERSYM])
    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betayx', 'betayz', 'betazy', 'deltay']

    def _fill_coefficients(self, c):
        c.deltay = c.deltax
        c.betayx = c.betaxy
        c.betayz = c.betaxz
        c.betazy = c.betazx
        super()._fill_coefficients(c)


class StencilCylinderDivFree3D(StencilCylinder3D, StencilDivFree3D):
    """Three dimensional stencil that is the same on two out of three axes and preserves :math:`\\operatorname{div}\\vec{B}=0`."""

    ### Note that the order of parent classes matters!
    ### MRO will call the inherited methods from left to right,
    ### setting delta_y = delta_x (Cylinder3D) first and than
    ### calculating the betas from the deltas (DivFree3D)
    Flags = StencilCylinder3D.Flags | StencilDivFree3D.Flags


class StencilIsotropic3D(StencilSymmetric3D):
    """Three dimensional stencil with symmetric :math:`\\beta_{ij}` that is the same on all axes."""

    Flags = StencilSymmetric3D.Flags | frozenset([StencilFlags.ISOTROPIC])
    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betaxz', 'betayz', 'deltay', 'deltaz']

    def _fill_coefficients(self, c):
        c.deltay = c.deltax
        c.deltaz = c.deltax
        c.betaxz = c.betaxy
        c.betayz = c.betaxy
        super()._fill_coefficients(c)


class StencilIsotropicDivFree3D(StencilIsotropic3D, StencilDivFree3D):
    """Three dimensional stencil with symmetric :math:`\\beta_{ij}` that is the same on all axes and preserves :math:`\\operatorname{div}\\vec{B}=0`."""

    ### Note that the order of parent classes matters!
    ### MRO will call the inherited methods from left to right,
    ### setting delta_z = delta_y = delta_x (Isotropic3D) first and than
    ### calculating the betas from the deltas (DivFree3D)
    Flags = StencilIsotropic3D.Flags | StencilDivFree3D.Flags




