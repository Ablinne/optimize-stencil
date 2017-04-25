
Read Me
=======

Synopsis
--------

This package is used to find optimal coefficients for extended stencils in a Maxwell solver.

Code Example
------------

    from extended_stencil import Optimize

    opt = Optimize(args)
    x, fmin = opt.optimize()
    coefficients = opt.stencil.coefficients(x)

Motivation
----------

Introducing extended stencils in a PIC simulation allows for an improved numerical dispersion relation in order to, e. g., suppress numerical Cherenkov radiation.
This introduces a lot of freedom - which can be used for various optimization targets..

Installation
------------

    git clone git@git.tpi.uni-jena.de:albn/optimize_stencil.git
    ./setup.py install --user

API Reference
-------------

Some more information will be found under https://git.tpi.uni-jena.de/albn/optimize_stencil/wikis/home

Tests
-----

    nosetests

Contributors
------------

If you have problems let me know (https://git.tpi.uni-jena.de/albn/optimize_stencil/issues).

<!-- ## License -->

<!--This project is released under the GPL.-->
