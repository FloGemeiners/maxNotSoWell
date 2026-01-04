"""
Collection of functions that allow for the construction of different scenarios to test the FEM field calculation with.

Usage:
    import demo_functions

Author:
    Florian Meiners - November 5, 2025; Last updated January 4, 2026

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

make_line_current(p0: Tuple[float, float], p1: Tuple[float, float], J0: float, thickness: float) \
        -> Callable[[float, float], Tuple[float, float]]:
    in-plane current density concentrated near a line segment (see below)
make_circular_current(center: Tuple[float, float], radius: float, J0: float, thickness: float) \
    -> Callable[[float, float], Tuple[float, float]]:
    in-plane current density concentrated near a circle (see below)
"""

import numpy as np
from typing import Tuple, Callable
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

def make_line_current(p0: Tuple[float, float], p1: Tuple[float, float], J0: float, thickness: float) \
        -> Callable[[float, float], Tuple[float, float]]:
    """
    Construct an in-plane current density J(x,y) concentrated near the line segment from p0 to p1.

    Parameters:
    -----------
    p0, p1 : (2,) tuples
        Endpoints of the line segment.
    J0 : float
        Magnitude of the current density along the line (per unit "thickness"). Direction is from p0 to p1.
    thickness : float
        Half-width of the strip around the line where the current is nonzero.
        Within |distance_perp| <= thickness the current is constant; outside it is zero.

    Returns:
    -----------
    J : callable (x, y) -> (Jx, Jy)
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    L_vec = p1 - p0
    L = np.linalg.norm(L_vec)
    if L == 0.0:
        raise ValueError("p0 and p1 must be distinct points")
    t_hat = L_vec / L  # line direction

    def J(x: float, y: float) -> Tuple[float, float]:
        pt = np.array([x, y], dtype=float)
        v = pt - p0

        s = np.dot(v, t_hat)
        if s < 0.0 or s > L:
            return (0.0, 0.0)  # outside segment

        d_perp = abs(v[0] * t_hat[1] - v[1] * t_hat[0])

        if d_perp > thickness:
            return (0.0, 0.0)  # outside the strip

        return (J0 * t_hat[0], J0 * t_hat[1])

    return J

def make_circular_current(center: Tuple[float, float], radius: float, J0: float, thickness: float) \
    -> Callable[[float, float], Tuple[float, float]]:
    """
    Construct an in-plane current density J(x,y) concentrated near a circle.

    Parameters:
    -----------
    center : (2,) tuple
        Center of the circle (xc, yc).
    radius : float
        Radius R of the circle.
    J0 : float
        Magnitude of the current density along the circle (inside the band).
        Direction is counterclockwise.
    thickness : float
        Half-width of the annular band where the current is nonzero:
        |rho - R| <= thickness.

    Returns:
    -----------
    J : callable (x, y) -> (Jx, Jy)
    """
    xc, yc = center

    def J(x: float, y: float) -> Tuple[float, float]:
        rx = x - xc
        ry = y - yc
        rho = np.hypot(rx, ry)  # sqrt(rx^2 + ry^2)

        if rho == 0.0:
            return (0.0, 0.0)
        if abs(rho - radius) > thickness:
            return (0.0, 0.0)

        rhat_x = rx / rho
        rhat_y = ry / rho

        that_x = -rhat_y
        that_y = rhat_x
        return (J0 * that_x, J0 * that_y)

    return J