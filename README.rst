
..  This is part of the Optimize Stencil Reference Manual.
    Copyright (c) 2017 Alexander Blinne, David Schinkel

Read Me
=======

Optimize Stencil is a python utility, consisting of one python package `extended_stencil` and two command line utilities.
It is used to calculate optimal coefficients for an extended stencil used in a Maxwell Solver.
More information can be found in the full documentation and our published research article.


Authors
-------

The original version of this program was written by David Schinkel <schinkel.d@posteo.de>.
The current version is maintained by Alexander Blinne <a.blinne@gsi.de>.


Requirements
------------

This utility/package requires Python 3 and SciPy/NumPy.


Recommended modules
-------------------

The module :mod:`psutil` can be optionally used to find the number of physical cores.


Installation
------------

The current version is available via https://github.com/Ablinne/optimize-stencil.
The package and command line utilites are installed via the usual setup.py invocation::

    git clone https://github.com/Ablinne/optimize-stencil
    ./setup.py install --user


Documentation & API Reference
-----------------------------

The full documentation is maintained using Sphinx and can be found at http://optimize-stencil.readthedocs.io/en/latest/. The corresponding research article can be found at https://arxiv.org/abs/1710.06829 and https://doi.org/10.1016/j.cpc.2017.10.010 .


Usage Example
-------------

The command line utilities are easy to use::

    optimize_stencil.py

    optimize_stencil.py --div-free --dtrange 0.6718 1.0

    calculate_omega.py --lehe --params 0.96


Code Example
------------

::

    from extended_stencil import Optimize

    opt = Optimize(args)
    x, fmin = opt.optimize()
    coefficients = opt.stencil.coefficients(x)


Tests
-----

Some unit tests are implemented and can be run using::

    nosetests


License
-------

This project is published under the GNU GENERAL PUBLIC LICENSE Version 3, see COPYING.



Contributors
------------

If you have problems let me know at https://github.com/Ablinne/optimize-stencil/issues.
