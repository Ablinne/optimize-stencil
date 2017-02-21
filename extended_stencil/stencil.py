
from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
from numpy.version import version as npver

npvmajor, npvminor, npvbugfix = [int(x) for x in npver.split('.')]

class namedarray:
    def __init__(self, field_names):
        self._fields = [field_name.strip() for field_name in field_names.split(',')]
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


class Stencil(metaclass = ABCMeta):
    Parameters = None
    Coefficients = None

    @property
    def dof(self):
        return len(self.Parameters._fields)

    @abstractmethod
    def coefficients(self, args): ...


Coefficients2D = namedarray("dt, alphax, alphay, betaxy, betayx, deltax, deltay")
Coefficients3D = namedarray("dt, alphax, alphay, alphaz, betaxy, betaxz, betayx, betayz, betazx, betazy, deltax, deltay, deltaz")

class StencilFree2D(Stencil):
    Coefficients = Coefficients2D
    Parameters = namedarray("dt, betaxy, betayx, deltax, deltay")

    def coefficients(self, args):
        c = self.Coefficients.convert(self.Parameters(args))
        c.alphax = 1.0 - 2.0 * c.betaxy - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 3.0 * c.deltay
        return c

class StencilFixed2D(Stencil):
    Coefficients = Coefficients2D
    Parameters = namedarray("dt, deltax, deltay")

    def coefficients(self, args):
        c = self.Coefficients.convert(self.Parameters(args))
        c.betaxy = c.deltay
        c.betayx = c.deltax
        c.alphax = 1.0 - 2.0 * c.betaxy - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 3.0 * c.deltay
        return c

    def args_ok(self, args):
        c = self.Coefficients.convert(self.Parameters(args))
        #print( 1.0/4.0 - (c.deltax+c.deltay))
        return 1.0/4.0 - (c.deltax+c.deltay)

class StencilFree3D(Stencil):
    Coefficients = Coefficients3D
    Parameters = namedarray("dt, betaxy, betaxz, betayx, betayz, betazx, betazy, deltax, deltay, deltaz")

    def coefficients(self, args):
        c = self.Coefficients.convert(self.Parameters(args))
        c.alphax = 1.0 - 2.0 * c.betaxy - 2.0 * c.betaxz - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 2.0 * c.betayz - 3.0 * c.deltay
        c.alphaz = 1.0 - 2.0 * c.betazx - 2.0 * c.betazy - 3.0 * c.deltaz
        return c

class StencilFixed3D(Stencil):
    Coefficients = Coefficients3D
    Parameters = namedarray("dt, deltax, deltay, deltaz")

    def coefficients(self, args):
        c = self.Coefficients.convert(self.Parameters(args))
        c.betaxy = c.deltay
        c.betaxz = c.deltaz
        c.betayx = c.deltax
        c.betayz = c.deltaz
        c.betazx = c.deltax
        c.betazy = c.deltay
        c.alphax = 1.0 - 2.0 * c.betaxy - 2.0 * c.betaxz - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 2.0 * c.betayz - 3.0 * c.deltay
        c.alphaz = 1.0 - 2.0 * c.betazx - 2.0 * c.betazy - 3.0 * c.deltaz
        return c

    def args_ok(self, args):
        c = self.Coefficients.convert(self.Parameters(args))
        #print( 1.0/6.0 - (c.deltax+c.deltay+c.deltaz))
        return 1.0/6.0 - (c.deltax+c.deltay+c.deltaz)

def get_stencil(args):
    if args.dim == 2:
        if args.div_free:
            return StencilFixed2D()
        else:
            return StencilFree2D()

    elif args.dim == 3:
        if args.div_free:
            return StencilFixed3D()
        else:
            return StencilFree3D()


