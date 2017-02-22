
import sys
import numpy as np
import scipy.optimize as scop
import itertools
import concurrent.futures as cf
import psutil

from .dispersion import Dispersion2D, Dispersion3D
from .stencil import get_stencil

class make_single_max_constraint:
    def __init__(self, index, maxval):
        self.index = index
        self.maxval = maxval

    def __call__(self, p):
        return self.maxval-p[self.index]


class make_single_min_constraint:
    def __init__(self, index, minval):
        self.index = index
        self.minval = minval

    def __call__(self, p):
        return p[self.index]-self.minval

class Optimize:
    def __init__(self, args):
        self.args = args
        self.constraints = []
        self.dim = args.dim
        self.div_free = args.div_free
        self.stencil = get_stencil(args)

        if self.dim == 2:
            self.dispersion = Dispersion2D(args.Y, args.Ngrid_low, self.stencil)
            if args.Ngrid_high != args.Ngrid_low:
                self.dispersion_high = Dispersion2D(args.Y, args.Ngrid_high, self.stencil)
            else:
                self.dispersion_high = self.dispersion
            Y = args.Y

        elif self.dim == 3:
            self.dispersion = Dispersion3D(args.Y, args.Z, args.Ngrid_low, self.stencil)
            if args.Ngrid_high != args.Ngrid_low:
                self.dispersion_high = Dispersion3D(args.Y, args.Z, args.Ngrid_high, self.stencil)
            else:
                self.dispersion_high = self.dispersion
            Y = args.Y
            Z = args.Z

        self.constraints.append( dict(type='ineq', fun=self.dispersion.stencil_ok) )
        self.constraints.append( dict(type='ineq', fun=self.dispersion.dt_ok) )

        for i, fname in enumerate(self.stencil.Parameters._fields):
            bounds = getattr(args, fname+'range')
            #print('constraint', i, fname, bounds)
            self.constraints.append( dict(type='ineq', fun=make_single_min_constraint(i, bounds[0])) )
            self.constraints.append( dict(type='ineq', fun=make_single_max_constraint(i, bounds[1])) )


    def _optimize_single(self, betadelta):
        x0 = [0, *betadelta]
        stencil_ok = self.dispersion.stencil_ok(x0)
        #print('x00={}, coefficients={}, stencil_ok(x0)={}'.format(x0, self.dispersion.coefficients, stencil_ok))
        if stencil_ok < 0:
            return None, float('inf')

        x0[0] = np.asscalar(stencil_ok*0.95)
        #print('x0', x0)
        res = scop.minimize(self.dispersion.norm, x0, method='SLSQP', constraints = self.constraints, options = dict(disp=False, iprint = 2))
        norm = self.dispersion_high.norm(res.x)
        print('x0={}, x={}, norm={}'.format(x0, res.x, norm))
        return res, norm

    def optimize(self):
        ranges_betadelta = []
        for fname in self.stencil.Parameters._fields[1:]:
            bounds = getattr(self.args, fname+'range')
            r = np.linspace(bounds[0], bounds[1], self.args.N+2)[1:-1]
            ranges_betadelta.append( r )
            #print('range', fname, r)

        bestnorm = None
        bestres = None

        nproc = psutil.cpu_count(logical=False)
        if psutil.cpu_count(logical=True) > nproc:
            nproc+=1

        if self.args.singlecore:
            mymap = map
        else:
            pool = cf.ProcessPoolExecutor(nproc)
            mymap = pool.map

        for res, norm in mymap(self._optimize_single, itertools.product(*ranges_betadelta)):
            if bestnorm is None or norm < bestnorm:
                bestnorm = norm
                bestres = res

        return bestres.x, bestnorm

    def omega_output(self, fname, x):
        return self.dispersion_high.omega_output(fname, x)
