
from . import minmax

import numpy as np
import scipy.interpolate as spinterp


from abc import ABCMeta, abstractmethod, abstractproperty


class Dispersion(metaclass = ABCMeta):
    #weight function
    w = 1.0
    dx = 1.0
    dim = 0
    dt_multiplier = 1.0
    stencil = None

    def __init__(self, runinit2=True):
        self._parameters = None
        self._sqrtarg = None
        self._sqrtres = None
        self._coefficients = None
        self.init2_run = False
        if runinit2:
            self.init2()

    def init2(self):
        self.init2_run = True

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if not self.init2_run:
            self.init2()

        if self._parameters is not None and np.all(parameters == self._parameters):
            #print('parameters unchanged', self._parameters, parameters)
            return

        # make sure we COPY the parameters! (np.array does that!)
        self._parameters = np.array(parameters)
        self._sqrtarg = None
        self._sqrtres = None
        self._coefficients = self.stencil.coefficients(parameters)
        #print('parameters changed to', self._parameters)

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def sqrtres(self):
        if self._sqrtres is None:
            self._sqrtres = np.sqrt(self.sqrtarg)

        if np.any(np.isnan(self._sqrtres)):
            print('This Function should not have been called')
            raise ValueError('Got NaNs in sqrt for params {}, stencil_ok {}'.format(self._parameters, self.stencil_ok(self._parameters)))
        return self._sqrtres

    @abstractproperty
    def sqrtarg():
        ...

    def dt_ok(self, parameters):
        self.parameters = parameters
        if np.any(np.isnan(parameters)):
            # in some weird cases the optimizer goes rogue and feeds me NaNs...
            return -1.

        c = self.coefficients
        stencil_ok = self.stencil_ok(parameters)

        if stencil_ok < 0:
            # these parameters are outside of the constraints
            return stencil_ok

        dx = self.dx
        dt_ok = self.dt_multiplier/np.max(dx*self.sqrtres) - c.dt

        if np.isnan(dt_ok):
            raise ValueError("dt_ok got NaN")

        return dt_ok

    def omega(self, parameters):
        self.parameters = parameters
        c = self.coefficients

        stencil_ok = self.stencil_ok(parameters)

        if stencil_ok < 0:
            # these parameters are outside of the constraints
            return None

        dt_ok = self.dt_ok(parameters)

        if dt_ok < 0 or c.dt <= 0:
            # these parameters are outside of the constraints
            return None

        dt = c.dt

        #set dx=1, everything is measured in units of dx
        dx = self.dx

        omega = (2./(dt*dx))*np.arcsin(dt*dx*self.sqrtres)

        if np.any(np.isnan(omega)):
            raise ValueError('Encountered NaNs in omega for params', parameters)

        return omega


    def omega_output(self, fname, parameters):
        omega = self.omega(parameters)
        kappa = self.kappamesh
        kmesh = self.kmesh
        k = self.k
        #print(omega, self.dks)
        vgs = np.gradient(omega, *self.dks, edge_order=2)
        vg = np.sqrt(sum(vgc**2 for vgc in vgs))
        vph = omega/k
        vph.ravel()[0] = 1.
        data = [*np.broadcast_arrays(*kappa), k, omega, vph, vg, *vgs]

        data = np.broadcast_arrays(*data)
        blocks = zip(*map(lambda x: np.vsplit(x, x.shape[0]), data))
        with open(fname, 'wb') as f:
            kappanames = ' '.join(['kx', 'ky', 'kz'][:len(kappa)])
            vgsnames = ' '.join(['vgx', 'vgy', 'vgz'][:len(vgs)])
            f.write(('#' + kappanames + ' k omega vph vg ' + vgsnames + '\n').encode('us-ascii'))

            for block_columns in blocks:
                np.savetxt(f, np.vstack(map(np.ravel, block_columns)).T)
                f.write(b'\n')


    def norm(self, parameters):
        if np.any(np.isnan(parameters)):
            # in some weird cases the optimizer goes rogue and feeds me NaNs...
            return 1e6

        omega = self.omega(parameters)
        if omega is None:
            # just make sure we do not return 'nan's if constraints are not met
            return 1e6

        #integrand & integration
        f = self.w*(omega - self.k)**2
        F = f.sum()
        F = F*(np.pi/(self.N-1))**self.dim

        return F

class Dispersion2D(Dispersion):
    dim = 2
    def __init__(self, Y, N, stencil, **kwargs):
        self.Y = Y
        self.N = N
        self.stencil = stencil
        super(self.__class__, self).__init__(**kwargs)


    def init2(self):
        x = np.linspace(0, np.pi, self.N)
        self.kappamesh = np.meshgrid(x, x, indexing='ij', sparse=True)
        self.kappax, self.kappay = self.kappamesh
        self.kx = self.kappax
        self.ky = self.kappay/self.Y
        self.dks = self.kx[1,0], self.ky[0,1]
        self.kmesh = self.kx, self.ky
        self.k = np.sqrt(self.kx**2 + self.ky**2)
        self.coskappax=np.cos(self.kappax)
        self.coskappay=np.cos(self.kappay)
        self.sx2 = 0.5*(1. - self.coskappax) #np.sin(kappax/2)**2
        self.sy2 = 0.5*(1. - self.coskappay) #np.sin(kappay/2)**2
        super().init2()


    def stencil_ok(self, parameters):
        self.parameters = parameters
        if np.any(np.isnan(parameters)):
            # in some weird cases the optimizer goes rogue and feeds me NaNs...
            return -1

        c = self.coefficients

        a = np.min(self.sqrtarg)

        b = [(c.alphay+2*c.betayx-c.deltay)/(self.Y**2),
                c.alphax-2*c.betaxy-c.deltax+(c.alphay-2*c.betayx-c.deltay)/(self.Y**2),
                c.alphax+2*c.betaxy-c.deltax
                ]

        if not a < 0:
            res = min(b)
        else:
            res = a

        if np.isnan(res):
            raise ValueError("stencil_ok got NaN")

        return res

    @property
    def sqrtarg(self):
        if self._sqrtarg is None:
            c = self.coefficients
            #set dx=1, everything is measured in units of dx
            dx = self.dx

            Ax = c.alphax + c.deltax*(1.0 + 2.0*self.coskappax) +2.*c.betaxy*self.coskappay
            Ay = c.alphay + c.deltay*(1.0 + 2.0*self.coskappay) +2.*c.betayx*self.coskappax

            self._sqrtarg = Ax*self.sx2/(dx**2) + Ay*self.sy2/((self.Y * dx)**2)
        return self._sqrtarg



class Dispersion3D(Dispersion):
    dim = 3
    def __init__(self, Y, Z, N, stencil):
        self.Y = Y
        self.Z = Z
        self.N = N
        self.stencil = stencil
        super(self.__class__, self).__init__()


    def init2(self):
        x = np.linspace(0, np.pi, self.N)
        self.kappamesh = np.meshgrid(x, x, x, indexing='ij', sparse=True)
        self.kappax, self.kappay, self.kappaz = self.kappamesh

        self.kx = self.kappax
        self.ky = self.kappay/self.Y
        self.kz = self.kappaz/self.Z
        self.kmesh = self.kx, self.ky, self.kz
        self.dks = self.kx[1,0,0], self.ky[0,1,0], self.kz[0,0,1]

        self.k = np.sqrt((self.kappax)**2 + (self.kappay/self.Y)**2 + (self.kappaz/self.Z)**2)
        self.coskappax=np.cos(self.kappax)
        self.coskappay=np.cos(self.kappay)
        self.coskappaz=np.cos(self.kappaz)
        self.sx2 = 0.5*(1. - self.coskappax) #np.sin(kappax/2)**2
        self.sy2 = 0.5*(1. - self.coskappay) #np.sin(kappay/2)**2
        self.sz2 = 0.5*(1. - self.coskappaz) #np.sin(kappaz/2)**2
        super().init2()


    def stencil_ok(self, parameters):
        self.parameters = parameters
        if np.any(np.isnan(parameters)):
            # in some weird cases the optimizer goes rogue and feeds me NaNs...
            return -1
        c = self.coefficients

        a = np.min(self.sqrtarg)

        b = [(c.alphaz + 2.0 * c.betazx + 2.0 * c.betazy - c.deltaz)/self.Z**2,
             (c.alphay + 2.0 * c.betayx + 2.0 * c.betayz - c.deltay)/self.Y**2,
             (c.alphay + 2.0 * c.betayx - 2.0 * c.betayz - c.deltay)/self.Y**2 + (c.alphaz + 2.0 * c.betazx - 2.0 * c.betazy - c.deltaz)/self.Z**2,
             c.alphax + 2.0 * c.betaxy + 2.0 * c.betaxz - c.deltax,
             c.alphax + 2.0 * c.betaxy - 2.0 * c.betaxz - c.deltax + (c.alphaz - 2.0 * c.betazx + 2.0 * c.betazy - c.deltaz)/self.Z**2,
             c.alphax - 2.0 * c.betaxy + 2.0 * c.betaxz - c.deltax + (c.alphay - 2.0 * c.betayx + 2.0 * c.betayz - c.deltay)/self.Y**2,
             c.alphax - 2.0 * c.betaxy - 2.0 * c.betaxz - c.deltax + (c.alphay - 2.0 * c.betayx - 2.0 * c.betayz - c.deltay)/self.Y**2 + (c.alphaz - 2.0 * c.betazx - 2.0 * c.betazy - c.deltaz)/self.Z**2]

        if not a < 0:
            res = min(b)
        else:
            res = a

        if np.isnan(res):
            raise ValueError("stencil_ok got NaN")

        return res

    @property
    def sqrtarg(self):
        if self._sqrtarg is None:
            c = self.coefficients
            #set dx=1, everything is measured in units of dx
            dx = self.dx

            Ax = c.alphax + c.deltax*(1.0 + 2.0*self.coskappax) +2.*c.betaxy*self.coskappay + 2.*c.betaxz*self.coskappaz
            Ay = c.alphay + c.deltay*(1.0 + 2.0*self.coskappay) +2.*c.betayx*self.coskappax + 2.*c.betayz*self.coskappaz
            Az = c.alphaz + c.deltaz*(1.0 + 2.0*self.coskappaz) +2.*c.betazx*self.coskappax + 2.*c.betazy*self.coskappay

            self._sqrtarg = Ax*self.sx2/(dx**2) + Ay*self.sy2/((self.Y*  dx)**2) + Az*self.sz2/((self.Z * dx)**2)
        return self._sqrtarg
