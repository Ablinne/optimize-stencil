#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import sys

import scipy.optimize as scop
import numpy as np
import argparse
import itertools

from extended_stencil import Optimize

def main():

    parser = argparse.ArgumentParser(description = "This script calculates the optimal coefficients for a FDTD stencil.")
    parser.add_argument("--N", default=3, type=int, help="Number of scan points for initial conditions in each coefficient (default: %(default)s).")
    parser.add_argument("--scan-dt", action='store_true', help="Scan also through the allowed range of the time step for initial conditions (default: %(default)s).")
    parser.add_argument("--Ngrid_low", default=50, type=int, help="Gridsize for the fast calculation of the norm in the simplex algorithm (default: %(default)s).")
    parser.add_argument("--Ngrid_high", default=200, type=int, help="Gridsize for the accurate norm calculation after simplex  run (default: %(default)s).")
    parser.add_argument("--dim", default=2, type=int, choices=[2,3], help="Dimension of the stencil to be optimized (default: %(default)s).")
    parser.add_argument('--div-free', dest='div_free', action='store_true', help="Constrain the derivative of div B == 0 (default: %(default)s).")
    parser.add_argument('--no-div-free', dest='div_free', action='store_false', help="Constrain the derivative of div B == 0 (default: %(default)s).")
    parser.set_defaults(div_free=True)
    parser.add_argument('--symmetric-axes', type=int, choices=[0,1,2], default=0, help="Define the count of axes of the stencil, that should be identical to another axis (default: %(default)s).")
    parser.add_argument('--symmetric', dest='symmetric_axes', action='store_const', const=1, help="Make sure the stencil has 1 symmetric axis, equivalent to --symmetric-axes 1")
    parser.add_argument('--no-symmetric', dest='symmetric_axes', action='store_const', const=0, help="Make sure all axes of the stencil are free, equivalent to --symmetric-axes 0")
    parser.add_argument('--symmetric-beta', action='store_true', help="Demand that the beta coefficients should form a symmetric matrix")
    parser.add_argument("--Y", default=1, type=float, help="Grid aspect ratio dy/dx (default: %(default)s).")
    parser.add_argument("--Z", default=1, type=float, help="Grid aspect ratio dz/dx (default: %(default)s).")
    parser.add_argument("--dt-multiplier", default=1.0, type=float, help="Multiplier to be applied to time step as returned by CFL condition (default: %(default)s).")
    parser.add_argument("--deltaxrange", default=[-1,0.25], nargs=2, type = float, metavar=('min', 'max'), help="Range of deltax (default: %(default)s).")
    parser.add_argument("--deltayrange", default=[-1,0.25], nargs=2, type = float, metavar=('min', 'max'), help="Range of deltay (default: %(default)s).")
    parser.add_argument("--deltazrange", default=[-1,0.25], nargs=2, type = float, metavar=('min', 'max'), help="Range of deltaz (default: %(default)s).")
    parser.add_argument("--betaxyrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betaxy (default: %(default)s).")
    parser.add_argument("--betaxzrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betaxz (default: %(default)s).")
    parser.add_argument("--betayxrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betayx (default: %(default)s).")
    parser.add_argument("--betayzrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betayz (default: %(default)s).")
    parser.add_argument("--betazxrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betazx (default: %(default)s).")
    parser.add_argument("--betazyrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betazy (default: %(default)s).")
    parser.add_argument("--dtrange", default=[0.1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of dt in units of dx (default: %(default)s).")
    #parser.add_argument("--part", default=[1,1], nargs=2, type=int, metavar=("i", "n"), help="Only perform part i of a total of n parts, this is useful for a parallelization without the need to communicate between the different kernels (default: %(default)s).")
    parser.add_argument('--singlecore', action='store_true')
    parser.add_argument("--output", default="standard", choices=["standard", "array", "epoch"], help="Output format, 'standard' prints a list of the named coefficients, 'array' prints the returned array x as it is with [dt, beta{xyz}{xyz}, delta{xyz}, norm] and 'epoch' prints it compatible with the input decks of the EPOCH-Code.")
    parser.add_argument("--write-omega", default = None, help="Write omega to file OUTFILE", metavar="OUTFILE")
    args = parser.parse_args()
    print(args)
    #sys.exit()

    opt = Optimize(args)
    x, fmin = opt.optimize()
    coefficients = opt.stencil.coefficients(x)
    #print(x)

    if(args.output=='standard'):
        print("norm=", fmin, "\n")
        print('stencil_ok=', opt.dispersion_high.stencil_ok(x))
        print('dt_ok=', opt.dispersion_high.dt_ok(x))
        for item in zip(opt.stencil.Coefficients._fields, coefficients[0]):
            print('{}={}'.format(*item))
    if(args.output=='array'):
        print([*coefficients[0], fmin])
    if(args.output=='epoch'):
        print("norm=", fmin, "\n")
        print("\tmaxwell_solver=free")
        for item in zip(opt.stencil.Coefficients._fields, coefficients[0]):
            print('\tstencil_{}={}'.format(*item))
    if args.write_omega:
        opt.omega_output(args.write_omega, x)

if __name__ == "__main__":
    main()
