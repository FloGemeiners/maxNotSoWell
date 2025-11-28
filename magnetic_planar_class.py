"""
Collection of classes allowing for the setup of 2D magnetostatic Maxwell problems. For now, sources of the magnetic
field are limited to being electric currents.
(to be understood as the magnetic pendant of the electrostatic case)

Usage:
    from magnetic_planar_class import *

Author:
    Florian Meiners - November 25, 2025; Last updated November 27, 2025
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Definition of a scalar field.
ScalarField = Callable[[float, float], float]

@dataclass
class MaterialMu:
    """
    Class to specify the distribution of permeability across the spatial domain.

    Attributes:
    -----------
    mu: Callable
        The value of the permeability at the given point.

    Methods:
    -----------
    at(self, x:float, y:float)
        Returns the permeability at the specified point.
    """
    mu: Callable[[float, float], float]
    def at(self, x: float, y: float) -> float:
        return self.mu(x, y)

@dataclass
class CurrentDensity:
    """
    Class to specify the distribution of current density across the spatial domain.

    Attributes:
    -----------
    j_vec: Callable
        The value of the current density in z direction at the given point.
        Note that the currents are necessarily perpendicular to the plane.

    Methods:
    ----------
    at(self, x:float, y:float)
        Returns the current density at the specified point.
    """
    j_vec: Callable[[float, float], float]
    def at(self, x: float, y: float) -> float:
        return self.j_vec(x, y)

@dataclass
class DirichletBCMagnetic:
    """
    Class to specify the Dirichlet boundary conditions for the magnetic field.

    Attributes:
    -----------
    nodes : np.ndarray
        The nodes to specify the position of the boundary.
    value : ScalarField
        See above.

    Methods:
    -----------
    values(self, XY: np.ndarray)
        Returns the values at the boundary.
    """
    nodes: np.ndarray
    value: ScalarField
    def values(self, XY: np.ndarray) -> np.ndarray:
        return np.array([self.value(x, y) for x, y in XY[self.nodes]], dtype=float)

class Magnetic2D:
    """
    Class to define a magnetic boundary value problem in two dimensions. The degree of freedom in this case is the z
    component of the magnetic vector potential.

    Attributes:
    -----------
        nodes : np.ndarray
            array of nodes to calculate the potential (and magnetic field) for
        tris : np.ndarray
            array of tris to use for the FEM analyis
        material : MaterialMu
            distribution of permeability across the spatial domain
        source : CurrentDensity
            distribution of current density across the spatial domain
        N : int
            number of nodes
        A_vec : np.ndarray
            magnetic vector potential (z component) at the nodes
        _assembled : boolean
            specifying whether the problem is completely assembled
        _K : np.ndarray
            stiffness matrix for specifying the FEM problem
        _f : np.ndarray
            load matrix for specifying the FEM problem

        In case boundary conditions are applied to the problem:
        _fixed_idx : np.ndarray (optional)
            indices specifying fixed nodes
        _free_idx : np.ndarray (optional)
            indices specifying free nodes
        _fixed_vals : np.ndarray (optional)
            specified fixed values
        _K_reduced : np.ndarray (optional)
            reduced _K-matrix
        _f_reduced : np.ndarray (optional)
            reduced _f-matrix
        _use_reduced : boolean (optional)
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
    def __init__(self, nodes: np.ndarray, tris:np.ndarray,
                 material: MaterialMu, source: Optional[CurrentDensity] = None):
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
        source: CurrentDensity, optional
            charge density across the spatial domain (default = None)
        """
        self.nodes = nodes
        self.tris = tris
        self.material = material
        self.source = source or CurrentDensity(lambda x, y: 0.0)
        self.N = nodes.shape[0]
        self.A_vec = np.zeros(self.N)
        self._assembled = False
        self._K = None
        self._f = None

    @staticmethod
    def _triangle_area_and_grads(p: np.ndarray) -> Tuple[float, np.ndarray]:
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

        TODO: Carefully check if this method is actually doing what it is supposed to in terms of correctly calculating gradients. Same holds for duplicates.
        """
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        area2 = np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        A = abs(area2) * 0.5
        b = np.array([y1 - y2, y2 - y0, y0 - y1], dtype=float)
        c = np.array([x2 - x1, x0 - x2, x1 - x0], dtype=float)
        grads = np.column_stack([b, c]) / (2.0 * A)
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
                Ke = mu * A * (grads @ grads.T)
                j_c = self.source.at(xc, yc)
                fe = np.full(3, j_c * A /3.0, dtype=float)
                for a in range(3):
                    ia = int(tri[a])
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
                Ke = mu * A * (grads @ grads.T)
                j_c = self.source.at(xc, yc)
                fe = np.full(3, j_c * A / 3.0, dtype=float)
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
        self: Magnetic2D
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

    def solve(self) -> np.ndarray:
        """
        Method for solving the magnetic FEM problem and returning the z component of the vector potential.

        Parameters:
        -----------
        self: Magnetic2D

        Returns:
        -----------
        self.A_vec
            the potential at the nodes

        Sets:
        -----------
        self.A_vec
        """
        import numpy as np
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            HAS_SCIPY = True
        except Exception:
            HAS_SCIPY = False

        if not self._assembled:
            self.assemble()

        if getattr(self, "_use_reduced", False):
            K = self._K_reduced
            f = self._f_reduced
            if HAS_SCIPY and sp.issparse(K):
                u_free = spla.spsolve(K, f)
            else:
                u_free = np.linalg.solve(K, f)

            A_vec_z = np.empty(self.N, dtype=float)
            A_vec_z[self._fixed_idx] = self._fixed_vals
            A_vec_z[self._free_idx] = u_free
            self.A_vec = A_vec_z
            return self.A_vec

        K_full, f_full = self._K, self._f
        if HAS_SCIPY and sp.issparse(K_full):
            self.A_vec = spla.spsolve(K_full, f_full)
        else:
            self.A_vec = np.linalg.solve(K_full, f_full)
        return self.A_vec

    def magnetic_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method for calculating the magnetic field.

        Parameters:
        -----------
        self: Magnetic2D

        Returns:
        -----------
        Bx_nodes
            x component of the magnetic field
        By_nodes
            y component of the magnetic field
        centers
            coordinates of the nodes
        """
        if self.A_vec is None or self.A_vec.size == 0:
            raise RuntimeError("Call solve() first.")
        N = self.N
        Bx_acc = np.zeros(N)
        By_acc = np.zeros(N)
        count = np.zeros(N)

        for tri in self.tris:
            p = self.nodes[tri]
            A, grads = self._triangle_area_and_grads(p)
            A_loc = self.A_vec[tri]
            test = grads
            B_elem = (A_loc[:, None] * grads).sum(axis=0)
            for n in tri:
                Bx_acc[n] += B_elem[1]
                By_acc[n] += -B_elem[0]
                count[n] += 1.0

        Bx_nodes = Bx_acc / np.maximum(count, 1.0)
        By_nodes = By_acc / np.maximum(count, 1.0)
        centers = self.nodes
        return Bx_nodes, By_nodes, centers
