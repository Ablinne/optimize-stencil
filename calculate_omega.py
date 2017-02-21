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
    parser.add_argument("--Ngrid_high", default=1000, type=int, help="Gridsize for the accurate norm calculation after simplex  run (default: %(default)s).")
    parser.add_argument("--dim", default=2, type=int, choices=[2,3], help="Dimension of the stencil to be optimized (default: %(default)s).")
    parser.add_argument('--div-free', dest='div_free', action='store_true', help="Constrain the derivative of div B == 0 (default: %(default)s).")
    parser.add_argument('--no-div-free', dest='div_free', action='store_false', help="Constrain the derivative of div B == 0 (default: %(default)s).")
    parser.set_defaults(div_free=True)
    parser.add_argument("--Y", default=1, type=float, help="Grid aspect ratio dy/dx (default: %(default)s).")
    parser.add_argument("--Z", default=1, type=float, help="Grid aspect ratio dz/dx (default: %(default)s).")
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


    stencil = get_stencil(args)

    if args.dim == 2:
        dispersion = Dispersion2D(args.Y, args.Ngrid_high, stencil)
    elif args.dim == 3:
        dispersion = Dispersion3D(args.Y, args.Z, args.Ngrid_high, stencil)

    if args.type == "free":
        x = args.params
    elif args.type == "yee":
        x = [0] * stencil.Parameters._nfields
        x[0] = 0.95 * dispersion.dt_ok(x)
    elif args.type == "pukhov":
        x = [0] * stencil.Parameters._nfields
        x[0] = 0.95
        for i, name in enumerate(stencil.Parameters._fields):
            if name.startswith('beta'):
                x[i] = 0.125
    elif args.type == "lehe":
        x = [0] * stencil.Parameters._nfields
        x[0] = 0.95
        for i, name in enumerate(stencil.Parameters._fields):
            if name.startswith('beta'):
                x[i] = 0.125
            if name == 'deltax':
                x[i] = -0.025


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
