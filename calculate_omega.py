#!/usr/bin/env python3
# -*- encoding: utf8 -*-

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

import sys

import scipy.optimize as scop
import numpy as np
import argparse
import itertools

from extended_stencil import *

def main():

    parser = argparse.ArgumentParser(description = "This script calculates the optimal coefficients for a FDTD stencil.")
    parser.add_argument("--Ngrid_high", default=200, type=int, help="Gridsize for the accurate norm calculation after simplex  run (default: %(default)s).")
    parser.add_argument("--dim", default=2, type=int, choices=[2,3], help="Dimension of the stencil to be optimized (default: %(default)s).")
    parser.add_argument('--div-free', action='store_true', help="Constrain the derivative of div B == 0 (default: %(default)s).")
    parser.add_argument('--symmetric-axes', type=int, choices=[0,1,2], default=0, help="Define the count of axes of the stencil, that should be identical to another axis (default: %(default)s).")
    parser.add_argument('--symmetric', dest='symmetric_axes', action='store_const', const=1, help="Make sure the stencil has 1 symmetric axis, equivalent to --symmetric-axes 1")
    parser.add_argument('--no-symmetric', dest='symmetric_axes', action='store_const', const=0, help="Make sure all axes of the stencil are free, equivalent to --symmetric-axes 0")
    parser.add_argument('--no-symmetric-beta', dest='symmetric_beta', action='store_false', help="Do not demand that the beta coefficients should form a symmetric matrix. Warning: this will introduce spurious dimensions into the optimization problem and thus lead to unreproducible results.")
    parser.add_argument("--Y", default=1, type=float, help="Grid aspect ratio dy/dx (default: %(default)s).")
    parser.add_argument("--Z", default=1, type=float, help="Grid aspect ratio dz/dx (default: %(default)s).")
    parser.add_argument("--dt-multiplier", default=1.0, type=float, help="Multiplier to be applied to time step as returned by CFL condition (default: %(default)s).")
    parser.add_argument("--params", nargs='*', type = float, help="Parameters for stencil.")
    parser.add_argument("--lehe", action='store_const', dest='type', const='lehe')
    parser.add_argument("--yee", action='store_const', dest='type', const='yee')
    parser.add_argument("--pukhov", action='store_const', dest='type', const='pukhov')
    parser.set_defaults(type="free")

    parser.add_argument("--weight", choices=['equal', 'xaxis', 'xaxis_soft'], default="equal", help="Choose weight function (default: %(default)s).")
    parser.add_argument("--weight-params", nargs='*', type = float, help="Parameters for weight function.", default = [])

    parser.add_argument("--output", default="standard", choices=["standard", "array", "epoch"], help="Output format, 'standard' prints a list of the named coefficients, 'array' prints the returned array x as it is with [dt, beta{xyz}{xyz}, delta{xyz}, norm] and 'epoch' prints it compatible with the input decks of the EPOCH-Code.")
    parser.add_argument("--write-omega", default = None, help="Write omega to file OUTFILE", metavar="OUTFILE")

    parser.add_argument("--physical-dx", type=float, help="Physical dx value for group/phase velocity calculations", default=None)
    parser.add_argument("--laser-wavelength", type=float, help="Calculate phase and group velocity for given wavelength", default=None)
    parser.add_argument("--laser-direction", nargs='*', type=float, help="Point laser in this direction (vector normalized automatically)", default=None)

    args = parser.parse_args()
    print(args)
    #sys.exit()

    non_div_free_stencils = ('lehe', 'pukhov')
    if args.type in non_div_free_stencils and args.div_free:
        print('Stencil of type {}, setting option --no-div-free'.format(non_div_free_stencils))
        args.div_free = False

    non_symmetric_stencils = ('lehe')
    if args.type in non_symmetric_stencils and args.symmetric_axes > 0:
        print('Stencil of type {}, setting option --no-symmetric'.format(non_symmetric_stencils))
        args.symmetric = False

    stencil = get_stencil(args)

    print('Selected stencil', stencil.__class__.__name__, 'requires parameters', stencil.Parameters._fields)

    if args.dim == 2:
        dispersion = Dispersion2D(args.Y, args.Ngrid_high, stencil)
    elif args.dim == 3:
        dispersion = Dispersion3D(args.Y, args.Z, args.Ngrid_high, stencil)

    if args.weight in weight_functions:
            set_weight = weight_functions[args.weight](*args.weight_params)
            set_weight(dispersion)

    dispersion.dt_multiplier = args.dt_multiplier

    if args.type == "free":
        x = args.params
    elif args.type == "yee":
        x = [0] * stencil.Parameters._nfields
        x[0] = np.asscalar(dispersion.dt_ok(x))

    elif args.type == "pukhov":
        x = [0] * stencil.Parameters._nfields
        for i, name in enumerate(stencil.Parameters._fields):
            if name.startswith('beta'):
                if name[5] == 'x':
                    x[i] = 0.125
                elif name[5] == 'y':
                    x[i] = 0.125/args.Y**2
                elif name[5] == 'z':
                    x[i] = 0.125/args.Z**2
        x[0] = np.asscalar(dispersion.dt_ok(x))


    elif args.type == "lehe":
        x = [0] * stencil.Parameters._nfields
        x[0] = args.params[0]
        for i, name in enumerate(stencil.Parameters._fields):
            if name.startswith('beta'):
                if name[5] == 'x':
                    x[i] = 0.125
                elif name[5] == 'y':
                    x[i] = 0.125/args.Y**2
                elif name[5] == 'z':
                    x[i] = 0.125/args.Z**2

            if name == 'deltax':
                x[i] = 0.25*(1-1/x[0]**2*np.sin(np.pi*x[0]/2.0))

    print('x=', x)
    print('stencil_ok=', dispersion.stencil_ok(x))
    print('dt_ok=', dispersion.dt_ok(x))
    fmin = dispersion.norm(x)
    coefficients = stencil.coefficients(x)
    #print(x)

    if(args.output=='standard'):
        print("norm=", fmin, "\n")
        for item in zip(stencil.Coefficients._fields, coefficients[0]):
            print('{}={}'.format(*item))
    if(args.output=='array'):
        print([*coefficients[0], fmin])
    if(args.output=='epoch'):
        print("norm=", fmin, "\n")
        print("\tmaxwell_solver=free")
        for item in zip(stencil.Coefficients._fields, coefficients[0]):
            print('\tstencil_{}={}'.format(*item))


    if args.laser_wavelength:
        if not args.physical_dx:
            print("To calculate dispersion at specified laser wavelength/k vector please give --physical-dx")
            sys.exit(1)


        if args.dim == 2:
            omega_spline = dispersion.omega_spline(x)

            def vph_vg(k):
                kx, ky = k
                kabs = np.sqrt(kx**2 + ky**2)
                vph = omega_spline(*k)/kabs

                vgx = omega_spline(*k, dx=1)
                vgy = omega_spline(*k, dy=1)
                vg = np.sqrt(vgx**2 + vgy**2)
                return map(np.asscalar, (vph, vg))

        if args.dim == 3:
            omega_interp, vg_interp = dispersion.omega_interp(x)

            def vph_vg(k):
                kx, ky, kz = k
                kabs = np.sqrt(kx**2 + ky**2 + kz**2)
                vph = omega_interp(k)/kabs

                vg = vg_interp(k)
                return map(np.asscalar, (vph, vg))

        kl = np.zeros(args.dim)
        if args.laser_direction:
            kl[:] = args.laser_direction
            kl /= np.sqrt(np.sum(kl*kl))
        else:
            kl[0] = 1
        kl *= 2*np.pi/args.laser_wavelength*args.physical_dx
        vph, vg = vph_vg(kl)
        print("Specified laser wavelength lambda={} dx leads to vph={} c and vg={} c.".format(args.laser_wavelength/args.physical_dx, vph, vg))


    if args.write_omega:
        dispersion.omega_output(args.write_omega, x)

if __name__ == "__main__":
    main()
