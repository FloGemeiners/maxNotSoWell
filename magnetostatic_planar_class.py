"""
Collection of classes allowing for the setup of 2D magnetostatic Maxwell problems. The source of the magnetic field is
the remanent flux density of a permanent magnet.

Usage:
    from magnetostatic_planar_class import *

Author:
    Florian Meiners - January 5, 2026
"""

from dataclasses import dataclass
from typing import Callable, Tuple
from magnetic_planar_class import MaterialMu
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Definitions of scalar and vector field
ScalarField = Callable[[float, float], float]
VectorField2D = Callable[[float, float], Tuple[float, float]]

@dataclass
class RemanentFluxDensityInPlane:
    """
    Class to specify the remanent flux density across the spatial domain. This is the source of the magnetic field.

    Attributes:
    -----------
    Br: Callable
        The value and direction of the remanent flux density at any given point.

    Methods:
    -----------
    at(self, x:float, y:float)
        Returns the remanent flux density at the specified point.
    """
    Br: VectorField2D  # returns (Brx, Bry)

    def at(self, x: float, y: float) -> np.ndarray:
        Brx, Bry = self.Br(x, y)
        return np.array([Brx, Bry], dtype=float)

class Magnetostatic2D:
    """
    Class to define a magnetostatic boundary value problem in two dimensions. The degree of freedom in this case is the
    (in this case) scalar potential phi which results from the remanent flux density.

    Attributes:
    -----------
        nodes: np.ndarray
            array of nodes to calculate the potential (and magnetic field) for
        tris: np.ndarray
            array of tris to use for the FEM analyis
        material: MaterialMu
            distribution of permeability across the spatial domain
        remanence: RemanentFluxDensityInPlane
            distribution and directions of the remanent flux density across the spatial domain
        N: int
            number of nodes
        phi: np.ndarray
            magnetic potential at the nodes
        _assembled: boolean
            specifying whether the problem is completely assembled
        _K: np.ndarray
            stiffness matrix for specifying the FEM problem
        _f: np.ndarray
            load matrix for specifying the FEM problem

        In case boundary conditions are applied to the problem:
        _fixed_idx: np.ndarray (optional)
            indices specifying fixed nodes
        _free_idx: np.ndarray (optional)
            indices specifying free nodes
        _fixed_vals: np.ndarray (optional)
            specified fixed values
        _K_reduced: np.ndarray (optional)
            reduced _K-matrix
        _f_reduced: np.ndarray (optional)
            reduced _f-matrix
        _use_reduced: boolean (optional)
            tells the solver whether to use the reduced matrices for calculation

        Methods:
        -----------
        _triangle_area_and_grads(p: np.ndarray)
            static method to return the areas and gradients at the specified triangle
        assemble(self)
            Assembles and returns the FEM problem
        apply_dirichlet(self, bcs)
            Applies the dirichlet boundary conditions to the magnetic problem.
        solve(self)
            Solves the FEM problem.
        magnetic_field(self)
            Computes the magnetic field at the nodes.
    """
    def __init__(self, nodes, tris, material, remanence: RemanentFluxDensityInPlane):
        """
        Constructor.

        Parameters:
        -----------
        nodes: np.ndarray
            array of nodes to solve the problem for
        tris: np.ndarray
            array of triangles to use for the FEM analyis
        material: MaterialMu
            distribution of permeability across the spatial domain
        source: RemanentFluxDensityInPlane, optional
            remanent flux density across the spatial domain (default = None)
        """
        self.nodes = nodes
        self.tris = tris
        self.material = material
        self.remanence = remanence

        self.N = nodes.shape[0]
        self.phi = np.zeros(self.N, dtype=float)

        self._assembled = False
        self._K = None
        self._f = None

    @staticmethod
    def _triangle_area_and_grads(p):
        """
        Method to return the areas and gradients of the specified triangle. This functions implicitly works with linear
        shape functions.

        Parameters:
        -----------
        p: np.ndarray
            array of points characterizing the triangle

        Returns:
        -----------
        A: np.ndarray
            areas of the triangle
        grads: np.ndarray
            gradients at the vertices of the triangle
        """
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        A = 0.5 * abs(area2)
        if A == 0.0:
            raise ValueError("Degenerate triangle with zero area.")

        b = np.array([y1 - y2, y2 - y0, y0 - y1], dtype=float)
        c = np.array([x2 - x1, x0 - x2, x1 - x0], dtype=float)
        grads = np.column_stack([b, c]) / (2.0 * A)  # (3,2): [∂x φ_i, ∂y φ_i]
        return A, grads

    def assemble(self):
        """
        Assembles the magnetic FEM problem by setting the stiffness and load matrices.

        Parameters:
        -----------
        self: Magnetic2D

        Returns:
        -----------
        Nothing

        Sets:
        -----------
        self._assembled, self._K and self._f
        """
        N = self.N
        if HAS_SCIPY:
            rows, cols, data = [], [], []
            f = np.zeros(N, dtype=float)

            for tri in self.tris:
                p = self.nodes[tri]
                A, grads = self._triangle_area_and_grads(p)
                xc, yc = p.mean(axis=0)

                mu = self.material.at(xc, yc)
                Br = self.remanence.at(xc, yc)

                Ke = mu * A * (grads @ grads.T)

                fe = np.array([A * float(np.dot(Br, grads[a])) for a in range(3)], dtype=float)

                for a in range(3):
                    ia = int(tri[a])
                    f[ia] += fe[a]
                    for b in range(3):
                        ib = int(tri[b])
                        rows.append(ia)
                        cols.append(ib)
                        data.append(Ke[a, b])

            K = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        else:
            K = np.zeros((N, N), dtype=float)
            f = np.zeros(N, dtype=float)

            for tri in self.tris:
                p = self.nodes[tri]
                A, grads = self._triangle_area_and_grads(p)
                xc, yc = p.mean(axis=0)

                mu = self.material.at(xc, yc)
                Br = self.remanence.at(xc, yc)

                Ke = mu * A * (grads @ grads.T)
                fe = np.array([A * float(np.dot(Br, grads[a])) for a in range(3)], dtype=float)

                for a in range(3):
                    ia = int(tri[a])
                    f[ia] += fe[a]
                    for b in range(3):
                        ib = int(tri[b])
                        K[ia, ib] += Ke[a, b]

        self._K, self._f = K, f
        self._assembled = True

    def apply_dirichlet(self, bcs):
        """
        Method for applying the Dirichlet boundary conditions (potential at boundary).
        When these boundary conditions are applied, the object receives additional attributes.

        Parameters:
        -----------
        self: Magnetostatic2D
        bcs: DirichletBCMagnetic

        Returns:
        -----------
        Nothing

        Sets:
        -----------
        self._K and self._f, as well as the optional
        _free_idx, _fixed_idx, _fixed_vals, _K_reduced, _f_reduced, _use_reduced
        """
        if not self._assembled:
            self.assemble()

        fixed = {}
        for bc in bcs:
            vals = bc.values(self.nodes)
            for n, v in zip(bc.nodes, vals):
                fixed[int(n)] = float(v)

        self._fixed_idx = np.array(sorted(fixed.keys()), dtype=int)
        self._fixed_vals = np.array([fixed[i] for i in self._fixed_idx], dtype=float)

        all_idx = np.arange(self.N, dtype=int)
        self._free_idx = np.setdiff1d(all_idx, self._fixed_idx, assume_unique=True)

        K = self._K
        f = self._f
        I = self._fixed_idx
        F = self._free_idx
        g = self._fixed_vals

        if HAS_SCIPY and sp.issparse(K):
            K = K.tocsr()
            K_ff = K[F, :][:, F]
            K_fc = K[F, :][:, I]
            f_f = f[F] - K_fc @ g
        else:
            K_ff = K[np.ix_(F, F)]
            K_fc = K[np.ix_(F, I)]
            f_f = f[F] - K_fc @ g

        self._K_reduced = K_ff
        self._f_reduced = f_f
        self._use_reduced = True

    def solve(self):
        """
        Method for solving the magnetic FEM problem and returning the magnetic potential.

        Parameters:
        -----------
        self: Magnetostatic2D

        Returns:
        -----------
        self.phi
            the potential at the nodes

        Sets:
        -----------
        self.phi
        """
        if not self._assembled:
            self.assemble()

        if getattr(self, "_use_reduced", False):
            K = self._K_reduced
            f = self._f_reduced
            if HAS_SCIPY and sp.issparse(K):
                u_free = spla.spsolve(K, f)
            else:
                u_free = np.linalg.solve(K, f)

            phi = np.empty(self.N, dtype=float)
            phi[self._fixed_idx] = self._fixed_vals
            phi[self._free_idx] = u_free
            self.phi = phi
            return self.phi

        K_full, f_full = self._K, self._f
        if HAS_SCIPY and sp.issparse(K_full):
            self.phi = spla.spsolve(K_full, f_full)
        else:
            self.phi = np.linalg.solve(K_full, f_full)
        return self.phi

    def magnetic_field(self):
        """
        Method for calculating the magnetic field.

        Parameters:
        -----------
        self: Magnetostatic2D

        Returns:
        -----------
        Bx_nodes
            x component of the magnetic field
        By_nodes
            y component of the magnetic field
        centers
            coordinates of the nodes
        """
        if self.phi is None or self.phi.size == 0:
            raise RuntimeError("Call solve() first.")

        N = self.N
        Bx_acc = np.zeros(N, dtype=float)
        By_acc = np.zeros(N, dtype=float)
        count = np.zeros(N, dtype=float)

        for tri in self.tris:
            p = self.nodes[tri]
            _, grads = self._triangle_area_and_grads(p)
            xc, yc = p.mean(axis=0)

            mu = self.material.at(xc, yc)
            Br = self.remanence.at(xc, yc)

            phi_loc = self.phi[tri]
            grad_phi = (phi_loc[:, None] * grads).sum(axis=0)  # [∂x phi, ∂y phi]

            B_elem = -mu * grad_phi + Br  # (2,)

            for n in tri:
                Bx_acc[n] += B_elem[0]
                By_acc[n] += B_elem[1]
                count[n] += 1.0

        Bx = Bx_acc / np.maximum(count, 1.0)
        By = By_acc / np.maximum(count, 1.0)
        return Bx, By, self.nodes
