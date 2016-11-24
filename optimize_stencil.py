#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import scipy.optimize as scop
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

#2D-Version
#-------------------------------------------------------------------------------
@_nperrchange(invalid='ignore')
def norm_omega_2d(x, Y, N, kappax, kappay, coskappax, coskappay):

    T=x[0]
    betaxy = x[1]
    betayx = x[2]
    deltax = x[3]
    deltay = x[4]
    #set dx=1, everything is measured in units of dx
    dx=1
    #weight function
    w=1
    #omega & k
    Ax = 1- 2*betaxy - 2*deltax + 2*deltax*coskappax + 2*betaxy*coskappay
    Ay = 1- 2*betayx - 2*deltay + 2*deltay*coskappay + 2*betayx*coskappax
    sx2 = 0.5*(1 - coskappax) #np.sin(kappax/2)**2
    sy2 = 0.5*(1 - coskappay) #np.sin(kappay/2)**2
    omega = (2/(T*dx))*np.arcsin(T*dx*np.sqrt(Ax*sx2/(dx**2) + Ay*sy2/(Y**2* dx**2)))
    k = np.sqrt((kappax)**2 + (kappay/Y)**2)
    #integrand & integration
    f = w*(omega - k)**2
    F = f.sum()
    F = F*(np.pi/(N-1))*(np.pi/(N-1))

    return F

def optimize_coefficients_2d(kappax, kappay, coskappax, coskappay, T, betaxy, betayx, deltax, deltay, Y=1, Ngrid=100):

    y, norm, _, _, _ = scop.fmin(norm_omega_2d, [T , betaxy, betayx, deltax, deltay], disp=False, full_output=True, args=(Y, Ngrid, kappax, kappay, coskappax, coskappay))
    return [*y, norm]

def search_coefficients_2d(N=3, Ngrid_low=100, Ngrid_high=1000, Y=1, deltaxrange=[-1,1], deltayrange=[-1,1], betaxyrange=[-1,1], betayxrange=[-1,1], Trange=[0.1,1], part=[1,1]):

    #fill return vector with yee values
    x1 = np.linspace(0, np.pi, Ngrid_high)
    x2 = np.linspace(0, np.pi, Ngrid_high)
    kappax, kappay = np.meshgrid(x1, x2)
    coskappax=np.cos(kappax)
    coskappay=np.cos(kappay)
    x=[0.95*Y/np.sqrt(1+Y**2),0,0,0,0,norm_omega_2d([0.95*Y/np.sqrt(1+Y**2), 0, 0, 0, 0], Y, Ngrid_high, kappax, kappay, coskappax, coskappay)]
    #activate progress bar, if possible
    try:
        from tqdm import tqdm
    except(ImportError):
        print('INFORMATION: install tqdm to see a progress bar.')
        #if tqdm not available set tqdm to unity
        tqdm = lambda x: x

    #construct kappax, kappay
    x1 = np.linspace(0, np.pi, Ngrid_low)
    x2 = np.linspace(0, np.pi, Ngrid_low)
    kappax_low, kappay_low = np.meshgrid(x1, x2)
    coskappax_low=np.cos(kappax_low)
    coskappay_low=np.cos(kappay_low)

    x1 = np.linspace(0, np.pi, Ngrid_high)
    x2 = np.linspace(0, np.pi, Ngrid_high)
    kappax_high, kappay_high = np.meshgrid(x1, x2)
    coskappax_high=np.cos(kappax_high)
    coskappay_high=np.cos(kappay_high)

    #construct tupels to be looped
    ranges = [
        np.linspace(deltayrange[0],deltayrange[1],N),
        np.linspace(deltaxrange[0],deltaxrange[1],N),
        np.linspace(betayxrange[0],betayxrange[1],N),
        np.linspace(betaxyrange[0],betaxyrange[1],N),
        np.linspace(Trange[0],Trange[1],N)
        ]
    #loop over coefficients and perform simplex
    l = len(list(itertools.product(*ranges)))
    looplist = list(itertools.product(*ranges))[(part[0]-1)*l//part[1]:part[0]*l//part[1]]
    for deltay ,deltax ,betayx ,betaxy ,T in tqdm(looplist):
        y = optimize_coefficients_2d(kappax_low, kappay_low, coskappax_low, coskappay_low, T , betaxy, betayx, deltax, deltay, Y, Ngrid_low)
        norm = norm_omega_2d(y[0:5], Y, Ngrid_high, kappax_high, kappay_high, coskappax_high, coskappay_high)
        if norm < x[5]:
            x = y
            x[5] = norm
    return x

#3D-Version
#-------------------------------------------------------------------------------
@_nperrchange(invalid='ignore')
def norm_omega_3d(x, Y, Z, N, kappax, kappay, kappaz, coskappax, coskappay, coskappaz):

    T=x[0]
    betaxy = x[1]
    betaxz = x[2]
    betayx = x[3]
    betayz = x[4]
    betazx = x[5]
    betazy = x[6]
    deltax = x[7]
    deltay = x[8]
    deltaz = x[9]
    #set dx=1, everything is measured in units of dx
    dx=1
    #weight function
    w=1
    #omega & k
    Ax = 1 - 2*betaxy - 2*betaxz - 2*deltax + 2*deltax*coskappax +2*betaxy*coskappay + 2*betaxz*coskappaz
    Ay = 1 - 2*betayx - 2*betayz - 2*deltay + 2*deltay*coskappay +2*betayx*coskappax + 2*betayz*coskappaz
    Az = 1 - 2*betazx - 2*betazy - 2*deltaz + 2*deltaz*coskappaz +2*betazx*coskappax + 2*betazy*coskappay
    sx2 = 0.5*(1 - coskappax) #np.sin(kappax/2)**2
    sy2 = 0.5*(1 - coskappay) #np.sin(kappay/2)**2
    sz2 = 0.5*(1 - coskappaz) #np.sin(kappaz/2)**2
    omega = (2/(T*dx))*np.arcsin(T*dx*np.sqrt(Ax*sx2/(dx**2) + Ay*sy2/(Y**2* dx**2) + Az*sz2/(Z**2 * dx**2)))
    #integrand & integration
    k = np.sqrt((kappax)**2 + (kappay/Y)**2 + (kappaz/Z)**2)
    f = w*(omega - k)**2
    F = f.sum()
    F = F*(np.pi/(N-1))*(np.pi/(N-1))*(np.pi/(N-1))

    return F

def optimize_coefficients_3d(kappax, kappay, kappaz, coskappax, coskappay, coskappaz, T , betaxy, betaxz, betayx, betayz, betazx, betazy, deltax, deltay, deltaz, Y=1, Z=1, Ngrid=100):

    y, norm, _, _, _ = scop.fmin(norm_omega_3d, [T , betaxy, betaxz, betayx, betayz, betazx, betazy, deltax, deltay, deltaz], disp=False, full_output=True, args=(Y, Z, Ngrid, kappax, kappay, kappaz, coskappax, coskappay, coskappaz))
    return [*y, norm]

def search_coefficients_3d(N=3, Ngrid_low=100, Ngrid_high=1000, Y=1, Z=1, deltaxrange=[-1,1], deltayrange=[-1,1], deltazrange=[-1,1], betaxyrange=[-1,1], betaxzrange=[-1,1], betayxrange=[-1,1], betayzrange=[-1,1], betazxrange=[-1,1], betazyrange=[-1,1], Trange=[0.1,1], part=[1,1]):

    #fill return vector with yee values
    x1 = np.linspace(0, np.pi, Ngrid_high)
    x2 = np.linspace(0, np.pi, Ngrid_high)
    x3 = np.linspace(0, np.pi, Ngrid_high)
    kappax, kappay, kappaz = np.meshgrid(x1, x2, x3)
    coskappax=np.cos(kappax)
    coskappay=np.cos(kappay)
    coskappaz=np.cos(kappaz)
    x=[0.95*Y*Z/np.sqrt(Y*Y + Y*Y*Z*Z + Z*Z),0,0,0,0,0,0,0,0,0,norm_omega_3d([0.95*Y*Z/np.sqrt(Y*Y + Y*Y*Z*Z + Z*Z), 0,0,0,0,0,0,0,0,0], Y, Z, Ngrid_high, kappax, kappay, kappaz, coskappax, coskappay, coskappaz)]
    #activate progress bar, if possible
    try:
        from tqdm import tqdm
    except(ImportError):
        print('INFORMATION: install tqdm to see a progress bar.')
        #if tqdm not available set tqdm to unity
        tqdm = lambda x: x

    #construct kappax, kappay
    x1 = np.linspace(0, np.pi, Ngrid_low)
    x2 = np.linspace(0, np.pi, Ngrid_low)
    x3 = np.linspace(0, np.pi, Ngrid_low)
    kappax_low, kappay_low, kappaz_low = np.meshgrid(x1, x2, x3)
    coskappax_low=np.cos(kappax_low)
    coskappay_low=np.cos(kappay_low)
    coskappaz_low=np.cos(kappaz_low)

    x1 = np.linspace(0, np.pi, Ngrid_high)
    x2 = np.linspace(0, np.pi, Ngrid_high)
    x3 = np.linspace(0, np.pi, Ngrid_high)
    kappax_high, kappay_high, kappaz_high = np.meshgrid(x1, x2, x3)
    coskappax_high=np.cos(kappax_high)
    coskappay_high=np.cos(kappay_high)
    coskappaz_high=np.cos(kappaz_high)
    #construct tupels to be looped
    ranges = [
        np.linspace(deltayrange[0],deltayrange[1],N),
        np.linspace(deltaxrange[0],deltaxrange[1],N),
        np.linspace(deltazrange[0],deltazrange[1],N),
        np.linspace(betazyrange[0],betayxrange[1],N),
        np.linspace(betazxrange[0],betayxrange[1],N),
        np.linspace(betayzrange[0],betayxrange[1],N),
        np.linspace(betayxrange[0],betayxrange[1],N),
        np.linspace(betaxzrange[0],betaxyrange[1],N),
        np.linspace(betaxyrange[0],betaxyrange[1],N),
        np.linspace(Trange[0],Trange[1],N)
        ]
    #loop over coefficients and perform simplex
    l = len(list(itertools.product(*ranges)))
    looplist = list(itertools.product(*ranges))[(part[0]-1)*l//part[1]:part[0]*l//part[1]]
    for deltay, deltax, deltaz, betazy, betazx, betayz, betayx, betaxz, betaxy, T in tqdm(looplist):
        y = optimize_coefficients_3d(kappax_low, kappay_low, kappaz_low, coskappax_low, coskappay_low, coskappaz_low, T , betaxy, betaxz, betayx, betayz, betazx, betazy, deltax, deltay, deltaz, Y, Z, Ngrid_low)
        norm = norm_omega_3d(y[0:10], Y, Z, Ngrid_high, kappax_high, kappay_high, kappaz_high, coskappax_high, coskappay_high, coskappaz_high)
        if norm < x[10]:
            x = y
            x[10] = norm
    return x

def main():

    #parse the arguments
    parser = argparse.ArgumentParser(description = "This script calculates the optimal coefficients for a FDTD stencil.")
    parser.add_argument("--N", default=3, type=int, help="Number of steps in each coefficient (default: %(default)s).")
    parser.add_argument("--Ngrid_low", default=100, type=int, help="Gridsize for the fast calculation of the norm in the simplex algorithm (default: %(default)s).")
    parser.add_argument("--Ngrid_high", default=1000, type=int, help="Gridsize for the accurate norm calculation after simplex  run (default: %(default)s).")
    parser.add_argument("--dim", default=2, type=int, choices=[2,3], help="Dimension of the stencil to be optimized (default: %(default)s).")
    parser.add_argument("--Y", default=1, type=float, help="Grid aspect ratio dy/dx (default: %(default)s).")
    parser.add_argument("--Z", default=1, type=float, help="Grid aspect ratio dz/dx (default: %(default)s).")
    parser.add_argument("--deltaxrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of deltax (default: %(default)s).")
    parser.add_argument("--deltayrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of deltay (default: %(default)s).")
    parser.add_argument("--deltazrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of deltaz (default: %(default)s).")
    parser.add_argument("--betaxyrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betaxy (default: %(default)s).")
    parser.add_argument("--betaxzrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betaxz (default: %(default)s).")
    parser.add_argument("--betayxrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betayx (default: %(default)s).")
    parser.add_argument("--betayzrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betayz (default: %(default)s).")
    parser.add_argument("--betazxrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betazx (default: %(default)s).")
    parser.add_argument("--betazyrange", default=[-1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of betazy (default: %(default)s).")
    parser.add_argument("--Trange", default=[0.1,1], nargs=2, type = float, metavar=('min', 'max'), help="Range of T=dt/dx (default: %(default)s).")
    parser.add_argument("--part", default=[1,1], nargs=2, type=int, metavar=("i", "n"), help="Only perform part i of a total of n parts, this is useful for a parallelization without the need to communicate between the different kernels (default: %(default)s).")
    parser.add_argument("--output", default="standard", choices=["standard", "array", "epoch"], help="Output format, 'standard' prints a list of the named coefficients, 'array' prints the returned array x as it is with [T, beta{xyz}{xyz}, delta{xyz}, norm] and 'epoch' prints it compatible with the input decks of the EPOCH-Code.")
    args = parser.parse_args()
    print(args)

    if args.dim==2:
        x = search_coefficients_2d(N=args.N, Ngrid_low=args.Ngrid_low, Ngrid_high=args.Ngrid_high, Y=args.Y, deltaxrange=args.deltaxrange, deltayrange=args.deltayrange, betaxyrange=args.betaxyrange, betayxrange=args.betayxrange, Trange=args.Trange, part=args.part)
        if(args.output=='standard'):
            print("norm=", x[-1], "\n")
            print("T=", x[0],"*dx/c", "\nbetaxy=", x[1], "\nbetayx=", x[2], "\ndeltax=", x[3], "\ndeltay=", x[4])
        if(args.output=='array'):
            print(x)
        if(args.output=='epoch'):
            print("norm=", x[-1], "\n")
            print("\tmaxwell_solver=free")
            print("\tstencil_dt=", x[0],"*dx/c", "\n\tstencil_betaxy=", x[1], "\n\tstencil_betayx=", x[2], "\n\tstencil_deltax=", x[3], "\n\tstencil_deltay=", x[4])
    if args.dim==3:
        x = search_coefficients_3d(N=args.N, Ngrid_low=args.Ngrid_low, Ngrid_high=args.Ngrid_high, Y=args.Y, Z=args.Z, deltaxrange=args.deltaxrange, deltayrange=args.deltayrange, deltazrange=args.deltazrange, betaxyrange=args.betaxyrange, betaxzrange=args.betaxzrange, betayxrange=args.betayxrange, betayzrange=args.betayzrange, betazxrange=args.betazxrange, betazyrange=args.betazyrange, Trange=args.Trange, part=args.part)
        if(args.output=='standard'):
            print("norm=", x[-1], "\n")
            print("T=", x[0], "\nbetaxy=", x[1], "\nbetaxz=", x[2], "\nbetayx=", x[3], "\nbetayz=", x[4], "\nbetazx=", x[5], "\nbetazy=", x[6], "\ndeltax=", x[7], "\ndeltay=", x[8], "\ndeltaz=", x[9])
        if(args.output=='array'):
            print(x)
        if(args.output=='epoch'):
            print("norm=", x[-1], "\n")
            print("\tmaxwell_solver=free")
            print("\tstencil_dt=", x[0], "\n\tstencil_betaxy=", x[1], "\n\tstencil_betaxz=", x[2], "\n\tstencil_betayx=", x[3], "\n\tstencil_betayz=", x[4], "\n\tstencil_betazx=", x[5], "\n\tstencil_betazy=", x[6], "\n\tstencil_deltax=", x[7], "\n\tstencil_deltay=", x[8], "\n\tstencil_deltaz=", x[9])

if __name__ == "__main__":
    main()
