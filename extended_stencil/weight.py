
import numpy as np

def weight_xaxis():
    """Factory function that returns a function `set_weight_xaxis` which
    acts on an object of :class:`extended_stencil.dispersion.Dispersion`
    and applies a weight function that only cares about the :math:`k_x`-axis"""
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
    """Factory function that returns a function `set_weight_xaxis_soft` which
    acts on an object of :class:`extended_stencil.dispersion.Dispersion`
    and applies a weight function that only cares about a region close to
    the :math:`k_x`-axis"""
    def set_weight_xaxis_soft(dispersion):
        if dispersion.dim == 2:
            dispersion.w = np.exp(-(dispersion.ky/width)**2)
        elif dispersion.dim == 3:
            dispersion.w = np.exp(-(dispersion.ky/width)**2-(dispersion.kz/width)**2)
        else:
            raise ValueError('Dispersion object not 2D or 3D')

    return set_weight_xaxis_soft

weight_functions = dict(xaxis = weight_xaxis, xaxis_soft = weight_xaxis_soft)
"""Dictionary that contains the factory functions for the different weights"""
