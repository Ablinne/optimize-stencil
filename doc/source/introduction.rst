
..  This is part of the Optimize Stencil Reference Manual.
    Copyright (c) 2017 Alexander Blinne, David Schinkel

Introduction
************

Motivation
----------

This software project aims at improving the properties of Maxwell Solvers using the Yee type staggered grid, for example within Particle-In-Cell simulations.
Introducing extended stencils in a PIC simulation allows for an improved numerical dispersion relation in order to, e. g., suppress numerical Cherenkov radiation.
The basics are thoroughly explained in our paper, that is in preparation to be published.
This introduces a lot of freedom - which can be used for various optimization targets.

Extended Stencils
-----------------
The Idea of extended stencils is about using not the simple two-point central difference stencil for the spatial derivatives in the Maxwell-Faraday equation, but using an extended stencil which takes additional grid points into account.
These additional stencil components are defined by a set of coefficients which have a direct influence on the numerical dispersion properties.
These coefficients underly, depending on a number of considerations, a number of preconditions.
Finding the best values for the remaining free coefficients is the task of this program.

Norm
----
To this end a norm function is built into this program, which aims for the best overall agreement of the grid dispersion relation with the free vacuum dispersion relation.
Other norms may have valid physical applications and may also be implemented in the future.

Parts of this codebase
----------------------
This codebase consists of four major components

* Package :ref:`extended_stencil <code>`
* Package :mod:`tests`
* Script :ref:`optimize_stencil.py <optimize-stencil-py>`
* Script :ref:`calculate_omega.py <calculate-omega-py>`

The package `extended_stencil` contains the implementation of the norm, of the dispersion relation and of some additional constraints for the coefficients of the extended stencil.

The module `tests` contains some unit tests for some of the classes defined in  `extended_stencil`.

The script `calculate_omega.py` is intended to be used to calculate the dispersion relation (the function :math:`\omega(\vec{k})`) and norm, given some specific coefficients.

The script `optimize_stencil.py` is used to find the optimal coefficients given a specific set of preconditions and constraints.
