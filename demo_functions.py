"""
Collection of functions that allow for the construction of different scenarios to test the FEM field calculation with.

Usage:
    import demo_functions

Author:
    Florian Meiners - November 5, 2025; Last updated January 5, 2026

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

_smooth_indicator(sd: float, smoothing: float):
    Helper function to obtain smooth outside curves in the horseshoe magnet.

_signed_distance_to_rect(x: float, y: float, hx: float, hy: float)
    Helper function to obtain the distance to a rectangle.

def make_rectangular_Br(center: Tuple[float, float], half_sizes: Tuple[float, float], Br0: float,
                        direction: Tuple[float, float] = (1.0, 0.0), angle_rad: float = 0.0, smoothing: float = 0.0) \
                        -> Callable[[float, float], Tuple[float, float]]:
    returns the in-plane shape of a rectangular magnet (see below)

make_horseshoe_Br(center: Tuple[float, float], leg_length: float, leg_thickness: float, gap: float,
                  yoke_thickness: float, Br0: float, angle_rad: float = 0.0, smoothing: float = 0.0,
                  opening: str = "up", gap_flux_lr: int = +1) -> VectorField2D:
    returns the in-plane shape of a horseshoe magnet (see below)
"""

import numpy as np
from typing import Tuple, Callable, Optional
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

VectorField2D = Callable[[float, float], Tuple[float, float]]

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
    center : (2, ) tuple
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

def _signed_distance_to_rect(x: float, y: float, hx: float, hy: float) -> float:
    dx = abs(x) - hx
    dy = abs(y) - hy
    outside = np.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside

def make_rectangular_Br(center: Tuple[float, float], half_sizes: Tuple[float, float], Br0: float,
                        direction: Tuple[float, float] = (1.0, 0.0), angle_rad: float = 0.0, smoothing: float = 0.0) \
                        -> Callable[[float, float], Tuple[float, float]]:
    """
    Returns the shape of a rectangular permanent magnet as an in-plane remanent flux density field Br(x,y).

    Parameters:
    -----------
    center : (2, ) tuple
        center of the rectangle (xc, yc)
    half_sizes : (2, ) tuple
        dimensions of the halves of the rectangle
    Br0 : float
        value of the remanent flux density inside the rectangle
    direction : (2, ) tuple
        directions of the field
    angle_rad : float
        rotation angle in radians
    smoothing : float
        smoothing factor to manipulate the roundness of curves

    Returns:
    -----------
    Br : callable (x, y) -> (Brx, Bry)
    """
    xc, yc = center
    hx, hy = half_sizes

    d = np.array(direction, dtype=float)
    nrm = np.linalg.norm(d)
    if nrm == 0.0:
        raise ValueError("direction must be nonzero")
    d /= nrm

    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))

    def Br(x: float, y: float) -> Tuple[float, float]:
        X = x - xc
        Y = y - yc

        xr =  c * X + s * Y
        yr = -s * X + c * Y

        sd = _signed_distance_to_rect(xr, yr, hx, hy)

        if smoothing <= 0.0:
            w = 1.0 if sd <= 0.0 else 0.0
        else:
            w = 0.5 * (1.0 - np.tanh(sd / smoothing))

        return (float(Br0 * w * d[0]), float(Br0 * w * d[1]))

    return Br

def _smooth_indicator(sd: float, smoothing: float) -> float:
    if smoothing <= 0.0:
        return 1.0 if sd <= 0.0 else 0.0
    return 0.5 * (1.0 - np.tanh(sd / smoothing))


def make_horseshoe_Br(center: Tuple[float, float], leg_length: float, leg_thickness: float, gap: float,
                      yoke_thickness: float, Br0: float, angle_rad: float = 0.0, smoothing: float = 0.0,
                      opening: str = "up", gap_flux_lr: int = +1) -> VectorField2D:
    """
    Returns the shape of a horseshoe permanent magnet as an in-plane remanent flux density field Br(x,y).

    Parameters:
    -----------
    center: (2, ) tuple
        center of the rectangle (xc, yc)
    leg_length: float
        length of the horseshoe leg
    leg_thickness: float
        thickness of the horseshoe leg
    gap: float
        distance between the horseshoe legs
    yoke_thickness: float
        thickness of the yoke connecting the legs
    Br0 : float
        value of the remanent flux density inside the horseshoe
    angle_rad : float
        rotation angle in radians
    smoothing : float
        smoothing factor to manipulate the roundness of curves
    opening : str
        direction in which the horseshoe is pointing
    gap_flux_lr: int
        direction of the flux between the legs

    Returns:
    -----------
    Br : callable (x, y) -> (Brx, Bry)
    """
    if opening not in ("up", "down"):
        raise ValueError("opening must be 'up' or 'down'")
    if gap_flux_lr not in (+1, -1):
        raise ValueError("gap_flux_lr must be +1 or -1")

    xc, yc = center
    t = float(leg_thickness)
    L = float(leg_length)
    g = float(gap)
    ty = float(yoke_thickness)

    outer_w = g + 2.0 * t
    total_h = ty + L

    if opening == "up":
        yoke_yc = -0.5 * total_h + 0.5 * ty
        legs_yc = yoke_yc + 0.5 * ty + 0.5 * L
        y_sign = +1.0
    else:
        yoke_yc = +0.5 * total_h - 0.5 * ty
        legs_yc = yoke_yc - 0.5 * ty - 0.5 * L
        y_sign = -1.0

    left_leg_xc  = -0.5 * g - 0.5 * t
    right_leg_xc = +0.5 * g + 0.5 * t

    leg_hx, leg_hy = 0.5 * t, 0.5 * L
    yoke_hx, yoke_hy = 0.5 * outer_w, 0.5 * ty

    left_dir  = np.array([0.0,  y_sign * float(gap_flux_lr)], dtype=float)
    right_dir = np.array([0.0, -y_sign * float(gap_flux_lr)], dtype=float)
    yoke_dir  = np.array([-float(gap_flux_lr), 0.0], dtype=float)

    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))

    def Br(x: float, y: float) -> Tuple[float, float]:
        X = x - xc
        Y = y - yc
        xl =  c * X + s * Y
        yl = -s * X + c * Y

        sd_left  = _signed_distance_to_rect(xl - left_leg_xc,  yl - legs_yc,  leg_hx,  leg_hy)
        sd_right = _signed_distance_to_rect(xl - right_leg_xc, yl - legs_yc,  leg_hx,  leg_hy)
        sd_yoke  = _signed_distance_to_rect(xl - 0.0,         yl - yoke_yc,  yoke_hx, yoke_hy)

        w_left  = _smooth_indicator(sd_left, smoothing)
        w_right = _smooth_indicator(sd_right, smoothing)
        w_yoke  = _smooth_indicator(sd_yoke, smoothing)

        w_occ = max(w_left, w_right, w_yoke)
        if w_occ <= 0.0:
            return (0.0, 0.0)

        d = w_left * left_dir + w_yoke * yoke_dir + w_right * right_dir
        n = float(np.linalg.norm(d))
        if n == 0.0:
            return (0.0, 0.0)
        d /= n

        Br_vec = Br0 * w_occ * d
        return (float(Br_vec[0]), float(Br_vec[1]))

    return Br
