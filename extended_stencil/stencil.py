
from collections import namedtuple
from enum import Enum
from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
from numpy.version import version as npver


npvmajor, npvminor, npvbugfix = [int(x) for x in npver.split('.')]

class namedarray:
    def __init__(self, field_names):
        if isinstance(field_names, str):
            self._fields = [field_name.strip() for field_name in field_names.split(',')]
        else:
            self._fields = field_names[:]
        self._nfields = len(self._fields)
        self.dtype = np.dtype([(field_name, float) for field_name in self._fields])

    def __call__(self, args):
        #print(self._fields, args)
        return np.asfarray(args).view(dtype = self.dtype).view(np.recarray)

    def convert(self, other):
        a = self(np.zeros(self._nfields))
        if npvminor >= 13:
            a[other.dtype.names] = other
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                a[:] = other
        return a

class StencilFlags(Enum):
    DIM2 = 0
    DIM3 = 1
    DIVFREE = 2
    ISOTROPIC = 3
    CYLINDERSYM = 4
    SYMMETRICBETA = 5

class Stencil(metaclass = ABCMeta):
    Parameters = None
    Coefficients = None
    stencils = {}

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
        return len(self._parameters)

    def coefficients(self, args):
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
        if sub is None:
            sub = cls
        for subclass in sub.__subclasses__():
            cls.register_subclasses(subclass)
            cls.stencils[subclass.Flags] = subclass

    @classmethod
    def get_stencil(cls, args):
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
    return Stencil.get_stencil(args)

class StencilFree2D(Stencil):
    Coefficients = namedarray("dt, alphax, alphay, betaxy, betayx, deltax, deltay")
    Flags = frozenset([StencilFlags.DIM2])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['alphax', 'alphay']

    def _fill_coefficients(self, c):
        c.alphax = 1.0 - 2.0 * c.betaxy - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 3.0 * c.deltay
        super()._fill_coefficients(c)


class StencilDivFree2D(StencilFree2D):
    Flags = StencilFree2D.Flags | frozenset([StencilFlags.DIVFREE])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betaxy', 'betayx']

    def _fill_coefficients(self, c):
        c.betaxy = c.deltay
        c.betayx = c.deltax
        super()._fill_coefficients(c)


class StencilSymmetric2D(StencilFree2D):
    Flags = StencilFree2D.Flags | frozenset([StencilFlags.SYMMETRICBETA])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betayx']

    def _fill_coefficients(self, c):
        c.betayx = c.betaxy
        super()._fill_coefficients(c)


class StencilIsotropic2D(StencilSymmetric2D):
    Flags = StencilSymmetric2D.Flags | frozenset([StencilFlags.ISOTROPIC])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['deltay']

    def _fill_coefficients(self, c):
        c.deltay = c.deltax
        super()._fill_coefficients(c)


class StencilIsotropicDivFree2D(StencilIsotropic2D, StencilDivFree2D):
    ### Note that the order of parent classes matters!
    ### MRO will call the inherited methods from left to right,
    ### setting delta_y = delta_x (Isotropic2D) first and than
    ### calculating the betas from the deltas (DivFree2D)

    Flags = StencilDivFree2D.Flags | StencilIsotropic2D.Flags



class StencilFree3D(Stencil):
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
    Flags = StencilFree3D.Flags | frozenset([StencilFlags.SYMMETRICBETA])

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betayx', 'betazx', 'betazy']

    def _fill_coefficients(self, c):
        c.betayx = c.betaxy
        c.betazx = c.betaxz
        c.betazy = c.betayz
        super()._fill_coefficients(c)


class StencilCylinder3D(StencilFree3D):
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
    ### Note that the order of parent classes matters!
    ### MRO will call the inherited methods from left to right,
    ### setting delta_y = delta_x (Cylinder3D) first and than
    ### calculating the betas from the deltas (DivFree3D)
    Flags = StencilCylinder3D.Flags | StencilDivFree3D.Flags


class StencilIsotropic3D(StencilSymmetric3D):
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
    ### Note that the order of parent classes matters!
    ### MRO will call the inherited methods from left to right,
    ### setting delta_z = delta_y = delta_x (Isotropic3D) first and than
    ### calculating the betas from the deltas (DivFree3D)
    Flags = StencilIsotropic3D.Flags | StencilDivFree3D.Flags




