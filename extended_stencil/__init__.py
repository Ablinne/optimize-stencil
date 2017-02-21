
from functools import wraps
import numpy as np


def minmax(arr):
    return np.min(arr), np.max(arr)

from extended_stencil.stencil import *
from extended_stencil.dispersion import *
from extended_stencil.optimize import *
