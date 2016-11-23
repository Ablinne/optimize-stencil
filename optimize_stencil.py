#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as scop
import time
import numexpr as ne
import numpy as np
import argparse
import itertools
from functools import wraps

def _nperrchange(**errchange):
    '''
    This decorator disables the numpy error handling for the given types as an
    argument. i.e. invalid='ignore' ignores all errors of the type invalid.
    '''
    def decorator(f):
        @wraps(f)
        def f_wrapper(*args, **kwargs):
            nperr = np.geterr()
            np.seterr(**errchange)
            ret = f(*args, **kwargs)
            np.seterr(**nperr)
            return ret
        return f_wrapper
    return decorator

@_nperrchange(invalid='ignore')
def omega_2d(kappax, kappay, dx, Y, T, betaxy, betayx, deltax, deltay):
    coskappax=np.cos(kappax)
    coskappay=np.cos(kappay)
    Ax = 1- 2*betaxy - 2*deltax + 2*deltax*coskappax + 2*betaxy*coskappay
    Ay = 1- 2*betayx - 2*deltay + 2*deltay*coskappay + 2*betayx*coskappax
    sx2 = 0.5*(1 - coskappax) #np.sin(kappax/2)**2
    sy2 = 0.5*(1 - coskappay) #np.sin(kappay/2)**2
    omega = (2/(T*dx))*np.arcsin(T*dx*np.sqrt(Ax*sx2/(dx**2) + Ay*sy2/(Y**2 * dx**2)))
    return omega

def norm_arg_2d(kappax, kappay, Y, T, betaxy, betayx, deltax, deltay):
    w = 1
    k = np.sqrt((kappax)**2 + (kappay/Y)**2)
    return w*(omega_2d(kappax, kappay, 1, Y, T, betaxy, betayx, deltax, deltay) - k)**2

def norm_omega_2d(x, Y, N):

    T=x[0]
    betaxy = x[1]
    betayx = x[2]
    deltax = x[3]
    deltay = x[4]

    # construct f(x,y) for given limits
    #-----------------------------------
    x = np.linspace(0, np.pi, N)
    y = np.linspace(0, np.pi, N)
    kappax, kappay = np.meshgrid(x, y)

    Nx, Ny = kappax.shape
    # construct 2-D integrand
    #-----------------------------------
    f = norm_arg_2d(kappax, kappay, Y, T, betaxy, betayx, deltax, deltay)
    F = f.sum()
    F = F*(np.pi/(Nx-1))*(np.pi/(Ny-1))

    return F

def optimize_coefficients_2d(T , betaxy, betayx, deltax, deltay, Y=1, Ngrid=100):
            y, norm, _, _, _ = scop.fmin(norm_omega_2d, [T , betaxy, betayx, deltax, deltay], disp=False, full_output=True, args=(Y, Ngrid))
            return [*y, norm]

def search_coefficients_2d(N=3, Ngrid_low=100, Ngrid_high=1000, Y=1, deltaxrange=[-1,1], deltayrange=[-1,1], betaxyrange=[-1,1], betayxrange=[-1,1], Trange=[0.1,1]):
    #fill return vector with yee values
    x=[0.65,0,0,0,0,norm_omega_2d([0.65, 0, 0, 0, 0], Y, Ngrid_high)]
    #activate progress bar, if possible
    try:
        from tqdm import tqdm
    except(ImportError):
        print('INFORMATION: install tqdm to see a progress bar.')
        #if tqdm not available set tqdm to unity
        tqdm = lambda x: x

    #construct tupels to be looped
    ranges = [
        np.linspace(deltayrange[0],deltayrange[1],N),
        np.linspace(deltaxrange[0],deltaxrange[1],N),
        np.linspace(betayxrange[0],betayxrange[1],N),
        np.linspace(betaxyrange[0],betaxyrange[1],N),
        np.linspace(Trange[0],Trange[1],N)
        ]
    #loop over coefficients and perform simplex
    for deltay ,deltax ,betayx ,betaxy ,T in tqdm(list(itertools.product(*ranges))):
        #print(deltay, " ", deltax, " ", betayx, " ", betaxy, " ", T )
        y = optimize_coefficients_2d(T , betaxy, betayx, deltax, deltay, Y, Ngrid_low)
        #y = scop.fmin(norm_omega, [T , betaxy, betayx, deltax, deltay], disp=False, args=(Y, Ngrid_low))
        norm = norm_omega_2d(y[0:5], Y, Ngrid_high)
        if norm < x[5]:
            x = y
            x[5] = norm
    return x

def main():
    #parse the arguments
    parser = argparse.ArgumentParser(description = "This script calculates the optimal coefficients for a FDTD stencil.")
    parser.add_argument("--N", default=3, type=int, help="Number of steps in each coefficient (default: %(default)s).")
    parser.add_argument("--Ngrid_low", default=100, type=int, help="Gridsize for the fast calculation of the norm in the simplex algorithm (default: %(default)s).")
    parser.add_argument("--Ngrid_high", default=1000, type=int, help="Gridsize for the accurate norm calculation after simplex  run (default: %(default)s).")
    parser.add_argument("--Y", default=1, help="Grid aspect ratio dy/dx (default: %(default)s).")
    parser.add_argument("--deltaxrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of deltax (default: %(default)s).")
    parser.add_argument("--deltayrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of deltay (default: %(default)s).")
    parser.add_argument("--betaxyrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betaxy (default: %(default)s).")
    parser.add_argument("--betayxrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betayx (default: %(default)s).")
    parser.add_argument("--Trange", default=[0.1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of T=dt/dx (default: %(default)s).")
    args = parser.parse_args()
    print(args)

    x = search_coefficients_2d(N=args.N, Ngrid_low=args.Ngrid_low, Ngrid_high=args.Ngrid_high, Y=args.Y, deltaxrange=args.deltaxrange, deltayrange=args.deltayrange, betaxyrange=args.betaxyrange, betayxrange=args.betayxrange, Trange=args.Trange)

    print("T=", x[0],"*dx/c", "\nbetaxy=", x[1], "\nbetayx=", x[2], "\ndeltax=", x[3], "\ndeltay=", x[4])

if __name__ == "__main__":
    main()
