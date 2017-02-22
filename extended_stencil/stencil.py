
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


class Stencil(metaclass = ABCMeta):
    Parameters = None
    Coefficients = None

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


class StencilFree2D(Stencil):
    Coefficients = namedarray("dt, alphax, alphay, betaxy, betayx, deltax, deltay")

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['alphax', 'alphay']

    def _fill_coefficients(self, c):
        c.alphax = 1.0 - 2.0 * c.betaxy - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 3.0 * c.deltay
        super()._fill_coefficients(c)

class StencilFixed2D(StencilFree2D):
    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['betaxy', 'betayx']

    def _fill_coefficients(self, c):
        c.betaxy = c.deltay
        c.betayx = c.deltax
        super()._fill_coefficients(c)


class StencilSymmetric2D(StencilFree2D):
    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['deltay', 'betaxy']

    def _fill_coefficients(self, c):
        c.deltay = c.deltax
        c.betaxy = c.betayx
        super()._fill_coefficients(c)


class StencilSymmetricFixed2D(StencilFixed2D, StencilSymmetric2D):
    pass


class StencilFree3D(Stencil):
    Coefficients = namedarray("dt, alphax, alphay, alphaz, betaxy, betaxz, betayx, betayz, betazx, betazy, deltax, deltay, deltaz")

    def _fixed_coefficients(self):
        return super()._fixed_coefficients() + ['alphax', 'alphay', 'alphaz']

    def _fill_coefficients(self, c):
        c.alphax = 1.0 - 2.0 * c.betaxy - 2.0 * c.betaxz - 3.0 * c.deltax
        c.alphay = 1.0 - 2.0 * c.betayx - 2.0 * c.betayz - 3.0 * c.deltay
        c.alphaz = 1.0 - 2.0 * c.betazx - 2.0 * c.betazy - 3.0 * c.deltaz
        super()._fill_coefficients(c)


class StencilFixed3D(StencilFree3D):
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


