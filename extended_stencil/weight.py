
import numpy as np

def weight_xaxis():
    def set_weight_xaxis(dispersion):
        dispersion.w = np.zeros_like(dispersion.k)
        if dispersion.dim == 2:
            dispersion.w[:,0] = 1.0
        elif dispersion.dim == 3:
            dispersion.w[:,0,0] = 1.0
        else:
            raise ValueError('Dispersion object not 2D or 3D')
    return set_weight_xaxis

def weight_xaxis_soft(width=0.1):
    def set_weight_xaxis_soft(dispersion):
        if dispersion.dim == 2:
            dispersion.w = np.exp(-(dispersion.ky/width)**2)
        elif dispersion.dim == 3:
            dispersion.w = np.exp(-(dispersion.ky/width)**2-(dispersion.kz/width)**2)
        else:
            raise ValueError('Dispersion object not 2D or 3D')

    return set_weight_xaxis_soft

weight_functions = dict(xaxis = weight_xaxis, xaxis_soft = weight_xaxis_soft)
