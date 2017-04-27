
# This file is part of the optimize_stencil project
#
# Copyright (c) 2017 Alexander Blinne, David Schinkel
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

import sys
import numpy as np
import scipy.optimize as scop
import itertools

from .dispersion import Dispersion, Dispersion2D, Dispersion3D
from .stencil import get_stencil

class make_single_max_constraint:
    """Class representing upper bounds for parameters."""

    def __init__(self, index, maxval):
        """:param index: Index of the parameter which is to be bounded
        :type index: int
        :param maxval: Upper bound
        :type maxval: float
        """
        self.index = index
        self.maxval = maxval

    def __call__(self, p):
        return self.maxval-p[self.index]


class make_single_min_constraint:
    """Class representing lower bounds for parameters."""

    def __init__(self, index, minval):
        """:param index: Index of the parameter which is to be bounded
        :type index: int
        :param maxval: Lower bound
        :type maxval: float
        """
        self.index = index
        self.minval = minval

    def __call__(self, p):
        return p[self.index]-self.minval

class Optimize:
    """Class that contains the main optimization logic."""

    def __init__(self, args):
        """:param args: Namespace object containing all the arguments.
        :type args: namespace"""
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
        self.dtmax = self.args.dtrange[1]

        for i, fname in enumerate(self.stencil.Parameters._fields):
            bounds = getattr(args, fname+'range')
            #print('constraint', i, fname, bounds)
            self.constraints.append( dict(type='ineq', fun=make_single_min_constraint(i, bounds[0])) )
            self.constraints.append( dict(type='ineq', fun=make_single_max_constraint(i, bounds[1])) )
            self.constraints_high.append( dict(type='ineq', fun=make_single_min_constraint(i, bounds[0])) )
            self.constraints_high.append( dict(type='ineq', fun=make_single_max_constraint(i, bounds[1])) )


    def _optimize_single(self, x0):
        x0 = list(x0)

        if x0[0] == None:
            x0[0] = 0
            dt_ok = np.asscalar(self.dispersion.dt_ok(x0))
            if dt_ok < 0:
                # Initial conditions violate constraints, reject
                return x0, None, float('inf')

            x0[0] = dt_ok
            x0[0] = min(x0[0], self.dtmax)
            x0[0] = max(x0[0], self.dtmin)

        x0 = np.asfarray(x0)

        stencil_ok = self.dispersion.stencil_ok(x0)
        if stencil_ok < 0:
            # Initial conditions violate constraints, reject
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
        """Execute the optimization.

        This will loop through the grid of starting values and return the best optimization result."""

        ranges_betadelta = []
        for fname in self.stencil.Parameters._fields[1:]:
            bounds = getattr(self.args, fname+'range')
            r = np.linspace(bounds[0], bounds[1], self.args.N+2)[1:-1]
            ranges_betadelta.append( r )
            #print('range', fname, r)

        if self.args.scan_dt:
            range_dt = np.linspace(self.dtmin, self.dtmax, self.args.N+2)[1:-1]
        else:
            range_dt = [None]

        bestnorm = None
        bestres = None
        bestx0 = None


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


        for x0, res, norm in mymap(self._optimize_single, itertools.product(range_dt, *ranges_betadelta)):
            print('x0={}, x={}, norm={}'.format(x0, res.x if res else None, norm))
            sys.stdout.flush()
            if bestnorm is None or norm < bestnorm:
                bestnorm = norm
                bestres = res
                bestx0 = x0


        if self.args.Ngrid_high != self.args.Ngrid_low:
            print()
            print('using this result as starting point for high-res optimization:')
            print('x0={},x={}, norm={}'.format(bestx0, bestres.x, bestnorm))
            bestres, bestnorm = self._optimize_final(bestres.x)

        return bestres.x, bestnorm

    def omega_output(self, fname, x):
        """Write resulting dispersion relation through :func:`extended_stencil.dispersion.Dispersion.omega_output`."""

        return self.dispersion_high.omega_output(fname, x)
