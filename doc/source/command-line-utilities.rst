
Documentation of command line utilities
***************************************

.. _optimize-stencil-py:

The Script `optimize_stencil.py`
================================

This script is used to optimize the coefficients of the extended stencil.
Command line arguments are used to input preconditions and constraints in order to find a stencil optimized for a specific purpose.

Usage examples
--------------

::

    `optimize_stencil.py`

Without any command line arguments, the script will use sensible defaults to calculate the coefficients of a 2D extended stencil.
These defaults result in an optimized stencil which is denoted `min. 1` in our paper.
Adding the option `--symmetric` renders this optimization process more efficient by using the fact that the grid is square and reducing the degrees of freedom.
In this case this does not change the result apart from numerical fluctuations.

::

    `optimize_stencil.py --div-free --dtrange 0.6718 1.0`

These options will force the script to obey Eq. (11) from out paper and set a lower bound of 0.6718 for the time step.
These settings will yield the optimized stencil `min. 2` from our paper.
The stencils `min. 3` and `min. 4` can be reproduced by adjusting the lower bound of the time step.

::

    `optimize_stencil.py --Y 10`

Find an optimal stencil for a grid with aspect ratio :math:`\frac{\Delta y}{\Delta x}=10`.
This will yield the optimized stencil presented for this grid in our paper.

::

    `optimize_stencil.py --dim 3 --symmetric-axes 2`

Find an optimal stencil in three dimensions for a uniform grid forcing identical behaviour for all axes

::

    ` optimize_stencil.py --dim 3 --Z 10 --Ngrid_low 30 --Ngrid_high 100`

Find an optimal stencil in three dimensions for a grid with non-square aspect ratio.
Also reduce number of grid points to speed up calculations.
This will take a few minutes even on a very fast machine.


.. _calculate-omega-py:

The Script `calculate_omega.py`
===============================

This script is used to calculate the phase and group velocities for a stencil with given parameters.
Some predefined stencils are available and parameters can also be input manually.

Usage examples
--------------

::

    `calculate_omega.py`

Without any command line arguments, the script has not enough input and exits with an error message.

::

    `calculate_omega.py --lehe --params 0.96`

These arguments will calculate the norm and dispersion relation for a stencil according to Lehes scheme using a time step of :math:`\frac{c\Delta t}{\Delta x}=0.96`.

::

    `calculate_omega.py --pukhov`

These arguments will calculate the norm and dispersion relation for a stencil according to Pukhovs NDFX scheme.

::

    `calculate_omega.py --yee`

These arguments will calculate the norm and dispersion relation for the Yee scheme.

