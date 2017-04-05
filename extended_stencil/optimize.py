
import sys
import numpy as np
import scipy.optimize as scop
import itertools

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
        self.constraints_high = []
        self.dim = args.dim
        self.div_free = args.div_free
        self.stencil = get_stencil(args)
        print('Using stencil', self.stencil.__class__.__name__, 'with free parameters', ", ".join(self.stencil.Parameters._fields))

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

        self.dispersion.dt_multiplier = args.dt_multiplier
        self.dispersion_high.dt_multiplier = args.dt_multiplier

        self.constraints.append( dict(type='ineq', fun=self.dispersion.stencil_ok) )
        self.constraints.append( dict(type='ineq', fun=self.dispersion.dt_ok) )
        self.constraints_high.append( dict(type='ineq', fun=self.dispersion_high.stencil_ok) )
        self.constraints_high.append( dict(type='ineq', fun=self.dispersion_high.dt_ok) )

        self.dtmin = self.args.dtrange[0]
        self.dtmax = self.args.dtrange[0]

        for i, fname in enumerate(self.stencil.Parameters._fields):
            bounds = getattr(args, fname+'range')
            #print('constraint', i, fname, bounds)
            self.constraints.append( dict(type='ineq', fun=make_single_min_constraint(i, bounds[0])) )
            self.constraints.append( dict(type='ineq', fun=make_single_max_constraint(i, bounds[1])) )
            self.constraints_high.append( dict(type='ineq', fun=make_single_min_constraint(i, bounds[0])) )
            self.constraints_high.append( dict(type='ineq', fun=make_single_max_constraint(i, bounds[1])) )


    def _optimize_single(self, betadelta):
        x0 = [0, *betadelta]
        dt_ok = np.asscalar(self.dispersion.dt_ok(x0))
        #print('x00={}, coefficients={}, stencil_ok(x0)={}'.format(x0, self.dispersion.coefficients, stencil_ok))
        if dt_ok < 0:
            # Initial conditions violate constraints, reject
            return x0, None, float('inf')

        x0[0] = dt_ok
        x0[0] = min(x0[0], self.dtmax)
        x0[0] = max(x0[0], self.dtmin)
        x0 = np.asfarray(x0)

        stencil_ok = self.dispersion.stencil_ok(x0)
        if stencil_ok < 0:
            return x0, None, float('inf')

        res = scop.minimize(self.dispersion.norm, x0, method='SLSQP', constraints = self.constraints, options = dict(disp=False, iprint = 2))
        norm = self.dispersion_high.norm(res.x)

        return x0, res, norm

    def _optimize_final(self, x0):
        res = scop.minimize(self.dispersion_high.norm, x0, method='SLSQP', constraints = self.constraints_high, options = dict(disp=False, iprint = 2))
        norm = res.fun
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


        if self.args.singlecore:
            mymap = map
        else:
            nproc = None
            try:
                import psutil
                nproc = psutil.cpu_count(logical=False)
                if psutil.cpu_count(logical=True) > nproc:
                    nproc+=1
            except ImportError:
                pass

            #import concurrent.futures as cf
            #pool = cf.ProcessPoolExecutor(nproc)
            #mymap = pool.map

            import multiprocessing as mp
            pool = mp.Pool(nproc)
            mymap = pool.imap

        for x0, res, norm in mymap(self._optimize_single, itertools.product(*ranges_betadelta)):
            print('x0={}, x={}, norm={}'.format(x0, res.x if res else None, norm))
            sys.stdout.flush()
            if bestnorm is None or norm < bestnorm:
                bestnorm = norm
                bestres = res

        res, norm = self._optimize_final(bestres.x)

        return res.x, norm

    def omega_output(self, fname, x):
        return self.dispersion_high.omega_output(fname, x)
