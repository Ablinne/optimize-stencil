#!/usr/bin/env python3
# -*- encoding: utf8 -*-

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
    parser.add_argument("--output", default="standard", choices=["standard", "array", "epoch"], help="Output format, 'standard' prints a list of the named coefficients, 'array' prints the returned array x as it is with [dt, beta{xyz}{xyz}, delta{xyz}, norm] and 'epoch' prints it compatible with the input decks of the EPOCH-Code.")
    parser.add_argument("--write-omega", default = None, help="Write omega to file OUTFILE", metavar="OUTFILE")
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
    if args.write_omega:
        dispersion.omega_output(args.write_omega, x)

if __name__ == "__main__":
    main()
