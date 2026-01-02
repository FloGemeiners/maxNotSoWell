"""
Collection of functions that allow for the construction of different scenarios to test the FEM field calculation with.

Usage:
    import demo_functions

Author:
    Florian Meiners - November 5, 2025; Last updated November 27, 2025

Functions:
-----------
rho_gauss(x0=0.3, y0=0.25, sigma=0.05, Q=1.0)
    provides a single two-dimensional positive charge distribution at (x_0,y_0) with a spread specified by sigma

rho_gauss_2(x0=0.3, y0=0.25, x1=0.7, y1=0.25, sigma=0.05, Q=1.0)
    like rho_gauss but with another negative charge distribution at (x_1,y_1) with the same spread

line_conductor(x0=0.5, y0=0.5)
    provides a single current source out of the plane at the specified location

line_conductor_2(x0=0.45, y0=0.45, x1=0.55, y1=0.55)
    provides two current sources (one out of, one into the plane) at the specified locations

mat_inhomogeneous(eps0 = 1.0, eps1 = 10.0, eps2 = 1.0, eps3 = 10.0, xsplit = 0.5, ysplit = 0.25)
    provides an inhomogeneous charge distribution, quartering the spatial domain into 4 quadrants separated by xsplit
    and ysplit; eps0 through eps3 are applied in the respective quadrants

mat_inhomogeneous_2(eps0 = 1.0, eps1 = 5.0, eps2 = 1.0, eps3 = 5.0, xsplit_1 = 0.25, xsplit_2 = 0.5, xsplit_3 = 0.75)
    like mat_inhomogeneous but with a separation of four areas in x direction

mat_inhomogeneous_mag(mu0 = 0.1, mu1 = 10.0, mu2 = 0.1, mu3 = 10.0, xsplit = 0.5, ysplit = 0.5)
    like mat_inhomogeneous_2 but with a permeability instead of a permittivity
"""

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def rho_gauss(x0=0.5, y0=0.25, sigma=0.001, Q=50.0):
    # simple function realizing a gaussian charge distribution in the plane
    inv2 = 1.0/(2*sigma*sigma)
    norm = Q/(2*np.pi*sigma*sigma)
    return lambda x, y: norm*np.exp(-((x-x0)**2 + (y-y0)**2)*inv2)


def rho_gauss_2(x0=0.3, y0=0.5, x1=0.7, y1=0.5, sigma=0.05, Q=50.0):
    # simple function realizing a bimodal gaussian charge distribution in the plane
    inv2 = 1.0/(2*sigma*sigma)
    norm = Q/(2*np.pi*sigma*sigma)
    return lambda x, y: + norm*np.exp(-((x-x0)**2 + (y-y0)**2)*inv2) - 2 * norm*np.exp(-((x-x1)**2 + (y-y1)**2)*inv2)


def line_conductor(x0=0.5, y0=0.5):
    # simple function realizing a single conductor with radius 0.1 out of the plane at (x0, y0)
    return lambda x, y: 20.0 if ((x - x0) ** 2 + (y - y0) ** 2 < 0.01) else 0.0


def line_conductor_2(x0=0.45, y0=0.45, x1=0.55, y1=0.55):
    # simple function realizing two conductors with radii 0.1 out of the plane at (x0, y0) and into the plane at (x1, y1)
    return lambda x, y: 40.0 if ((x - x0) ** 2 + (y - y0) ** 2 < 0.01) else (-40 if ((x - x1) ** 2 + (y - y1) ** 2 < 0.01) else 0.0)


def mat_inhomogeneous(eps0 = 1.0, eps1 = 10.0, eps2 = 1.0, eps3 = 10.0, xsplit = 0.5, ysplit = 0.5):
    # simple function realizing an inhomogeneous material distribution in the plane (division into 4 quadrants)
    return lambda x, y: eps0 if (x <= xsplit and y <= ysplit) else (eps1 if (x <= xsplit and y > ysplit) else
                                                                  (eps2 if (x > xsplit and y <= ysplit) else eps3))


def mat_inhomogeneous_2(eps0 = 1.0, eps1 = 5.0, eps2 = 1.0, eps3 = 5.0,
                        xsplit_1 = 0.25, xsplit_2 = 0.5, xsplit_3 = 0.75):
    # simple function realizing an inhomogeneous material distribution in the plane (division into 4 parallel sections)
    return lambda x, y: eps0 if (x <= xsplit_1) else (eps1 if (x <= xsplit_2) else (eps2 if (x <= xsplit_3) else eps3))


def mat_inhomogeneous_mag(mu0 = 0.1, mu1 = 10.0, mu2 = 0.1, mu3 = 10.0, xsplit = 0.5, ysplit = 0.5):
    # simple function realizing an inhomogeneous material distribution in the plane (division into 4 quadrants)
    return lambda x, y: mu0 if (x <= xsplit and y <= ysplit) else (mu1 if (x <= xsplit and y > ysplit) else
                                                                  (mu2 if (x > xsplit and y <= ysplit) else mu3))

def j_inplane(x: float, y: float) -> tuple[float, float]:
    # Simple example: constant Jx in some region
    if 0.25 <= x <= 0.75 and 0.25 <= y <= 0.75:
        return (1.0, 0.0)
    else:
        return (0.0, 0.0)
