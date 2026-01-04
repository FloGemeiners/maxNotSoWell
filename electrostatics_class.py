"""
Collection of classes allowing for the setup of 2D electrostatic Maxwell problems.

Usage:
    from electrostatics_class import *

Author:
    Florian Meiners - November 5, 2025; Last updated January 2, 2026
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import numpy as np

from finite_element_classes import *

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# note to myself... Definition of a scalar field.
ScalarField = Callable[[float, float], float]

@dataclass
class MaterialEpsilon:
    """
    Class to specify the distribution of permittivity across the spatial domain.

    Attributes:
    -----------
    epsilon: Callable
        The value of the permittivity at the given point.

    Methods:
    -----------
    at(self, x:float, y:float)
        Returns the permittivity at the specified point.
    """
    epsilon: Callable[[float, float], float]
    def at(self, x: float, y: float) -> float:
        return float(self.epsilon(x, y))

@dataclass
class ChargeDensity:
    """
    Class to specify the distribution of charge density across the spatial domain.

    Attributes:
    -----------
    rho: Callable
        The value of the charge density at the given point.

    Methods:
    -----------
    at(self, x:float, y:float)
        Returns the charge density at the specified point.
    """
    rho: ScalarField
    def at(self, x: float, y: float) -> float:
        return float(self.rho(x, y))

@dataclass
class DirichletBC:
    """
    Class to specify the Dirichlet boundary conditions for the electrostatic problem.

    Attributes:
    -----------
    nodes: np.ndarray
        The nodes to specify the position of the boundary.
    value: ScalarField
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

class TriangleQuadrature:
    """
    Helper class to facilitate a simple Gaussian quadrature rule on the triangle.
    (In the previous implementation, this was provided by the _triangle_area_and_grads method.)

    Attributes:
    -----------
    points: np.ndarray
        Centroid of the reference triangle.
    weights: np.ndarray
        Area of the reference triangle.
    """

    def __init__(self):
        self.points = np.array([[1.0/3.0, 1.0/3.0]])
        self.weights = np.array([[0.5]])

class Electrostatics2D:
    """
    Class to define an electrostatic boundary value problem in two dimensions. The degree of freedom in this case is
    the electric potential.

    Attributes:
    -----------
    mesh: Mesh2DRect
        The mesh across the spatial domain in which to solve the boundary value problem.
    nodes: np.ndarray
        array of nodes to calculate the potential (and electric field) for
    tris: np.ndarray
        array of tris to use for the FEM analysis
    material: MaterialEpsilon
        distribution of permittivity across the spatial domain
    source: ChargeDensity
        distribution of charge density across the spatial domain
    N: int
        number of nodes
    phi: np.ndarray
        potential at the nodes
    fin_elem: LagrangeP1Tri
        Instance of the reference finite element that the method is based on
    _assembled: boolean
        specifying whether the problem is completely assembled
    _K: np.ndarray
        stiffness matrix for specifying the FEM problem
    _f: np.ndarray
        load matrix for specifying the FEM problem

    In case boundary conditions are applied to the problem:
    -------------------------------------------------------
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
    ----------
    _triangle_area_and_grads(p: np.ndarray)
        static method to return the areas and gradients at the specified triangle
    _element_matrices_from_fin_elem(self, fin_elem, cell_vertices, quad)
        Essentially the same as _triangle_area_and_grads, but retrieves the information directly from an element.
    assemble(self)
        Assembles the FEM problem.
    assemble_by_elements(self)
        Like assemble() but based on element-wise computations.
    apply_dirichlet(self, bcs)
        Applies the dirichlet boundary condition to the electrostatic problem.
    solve(self)
        Solves the FEM problem.
    electric_field(self)
        Computes the electric field at the nodes.
    """
    def __init__(self, mesh: np.ndarray, material: MaterialEpsilon, source: Optional[ChargeDensity] = None,
                 fin_elem=None):
        """
        Constructor.

        Parameters:
        -----------
        mesh: Mesh2DRect
            mesh across the spatial domain
        nodes: np.ndarray
            array of nodes to solve the problem for
        tris: np.ndarray
            array of triangles to use for the FEM analysis
        material: MaterialEpsilon
            distribution of permittivity across the spatial domain
        source: ChargeDensity, optional
            charge density across the spatial domain (default = None)
        """
        self.mesh = mesh
        self.nodes = mesh.nodes
        self.tris = mesh.tris
        self.material = material
        self.source = source or ChargeDensity(lambda x, y: 0.0)
        # this is so that the Electrostatics2D class knows what type of element it is using:
        self.fin_elem = fin_elem if fin_elem is not None else LagrangeP1Tri()
        self.N = self.nodes.shape[0]
        self.phi = np.zeros(self.N)
        self._assembled = False
        self._K = None
        self._f = None

    @staticmethod
    def _triangle_area_and_grads(p: np.ndarray) -> Tuple[float, np.ndarray]:
        # Leading underscores indicate that a function, variable, ... is meant for internal use
        # (or private, for that matter).
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
        area2 = np.abs((x0 - x2) * (y1 - y0) - (x0 - x1) * (y2 - y0))
        A = abs(area2) * 0.5
        b = np.array([y1 - y2, y2 - y0, y0 - y1], dtype=float)
        c = np.array([x2 - x1, x0 - x2, x1 - x0], dtype=float)
        grads = np.column_stack([b, c]) / (2.0 * A)
        return A, grads

    def _element_matrices_from_fin_elem(self, fin_elem, cell_vertices, quad) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the local stiffness matrix Ke and load vector fe for one element represented by an FiniteElement object.

        Parameters:
        -----------
        self: Electrostatics2D
        fin_elem: FiniteElement
        cell_vertices: np.ndarray
        quad: TriangleQuadrature

        Returns:
        -----------
        Ke: np.ndarray
            local stiffness matrix
        fe_local: np.ndarray
            local load vector
        """
        n_dofs = fin_elem.num_local_dofs()
        Ke = np.zeros((n_dofs, n_dofs), dtype=float)
        fe_local = np.zeros(n_dofs, dtype=float)

        # geometry
        J, detJ, invJT, x0 = fin_elem._compute_affine_jacobian(cell_vertices)

        # physical quadrature weights
        w_phys = quad.weights * abs(detJ)

        # FE basis / gradients on physical cell
        phi = fin_elem.evaluate_basis(cell_vertices, quad.points)
        grads = fin_elem.evaluate_gradients(cell_vertices, quad.points)

        for q in range(quad.points.shape[0]):
            xi_eta = quad.points[q]
            x_q = x0 + J @ xi_eta  # physical quad point
            wq = w_phys[q]

            eps = self.material.at(x_q[0], x_q[1])
            rho = self.source.at(x_q[0], x_q[1])

            grad_q = grads[q]
            phi_q = phi[q]

            # stiffness
            for a in range(n_dofs):
                for b in range(n_dofs):
                    Ke[a, b] += eps * np.dot(grad_q[a], grad_q[b]) * wq

            # load
            for a in range(n_dofs):
                fe_local[a] += rho * phi_q[a] * wq

        return Ke, fe_local

    def assemble(self):
        """
        Assembles the electrostatic FEM problem by setting the stiffness and load matrices.

        Parameters:
        -----------
        self: Electrostatics2D

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
                eps = self.material.at(xc, yc)
                # capacitance of the material in the triangle (corresponds to stiffness mechanically)
                Ke = eps * A * (grads @ grads.T)
                rho_c = self.source.at(xc, yc)
                fe = np.full(3, rho_c * A / 3.0, dtype=float)  # charge distribution across material
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
                eps = self.material.at(xc, yc)
                Ke = eps * A * (grads @ grads.T)
                rho_c = self.source.at(xc, yc)
                fe = np.full(3, rho_c * A / 3.0, dtype=float)
                for a in range(3):
                    ia = int(tri[a])
                    f[ia] += fe[a]
                    for b in range(3):
                        ib = int(tri[b])
                        K[ia, ib] += Ke[a, b]
        self._K, self._f = K, f
        self._assembled = True

    def assemble_by_elements(self):
        """
        Assembles the electrostatic FEM problem using a FiniteElement object and element-wise contributions. Like the
        'assemble()' Method, sets the K and f attributes. This reproduces the old assemble() behaviour for P1 Lagrange
        with 1-point quadrature.

        Parameters:
        -----------
        self: Electrostatics2D

        Returns:
        -----------
        Nothing

        Sets:
        -----------
        self._assembled, self._K and self._f
        """
        N = self.N
        fin_elem = self.fin_elem
        quad = TriangleQuadrature()

        if HAS_SCIPY:
            rows, cols, data = [], [], []
            f = np.zeros(N, dtype=float)

            for tri in self.tris:
                cell_vertices = self.nodes[tri]  # (3,2)
                Ke, fe_local = self._element_matrices_from_fin_elem(fin_elem, cell_vertices, quad)

                # DOF mapping: for P1 H1 elements the DOFs are the  node indices
                dofs = tri

                for a, Ia in enumerate(dofs):
                    Ia = int(Ia)
                    f[Ia] += fe_local[a]
                    for b, Ib in enumerate(dofs):
                        Ib = int(Ib)
                        rows.append(Ia)
                        cols.append(Ib)
                        data.append(Ke[a, b])

            K = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

        else:
            K = np.zeros((N, N), dtype=float)
            f = np.zeros(N, dtype=float)

            for tri in self.tris:
                cell_vertices = self.nodes[tri]
                Ke, fe_local = self._element_matrices_from_fin_elem(fe, cell_vertices, quad)
                dofs = tri

                for a, Ia in enumerate(dofs):
                    Ia = int(Ia)
                    f[Ia] += fe_local[a]
                    for b, Ib in enumerate(dofs):
                        Ib = int(Ib)
                        K[Ia, Ib] += Ke[a, b]

        self._K, self._f = K, f
        self._assembled = True

    def apply_dirichlet(self, bcs):
        """
        Method for applying the Dirichlet boundary conditions (potential at boundary).
        When these boundary conditions are applied, the object receives additional attributes.

        Parameters:
        -----------
        self: Electrostatics2D
        bcs: DirichletBC

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
        Method for solving the electrostatic FEM problem and returning the potential.

        Parameters:
        -----------
        self: Electrostatics2D

        Returns:
        -----------
        self.phi
            the potential at the nodes

        Sets:
        -----------
        self.phi
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

    def electric_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method for calculating the electric field.

        Parameters:
        -----------
        self: Electrostatics2D

        Returns:
        -----------
        Ex_nodes
            x component of the electric field
        Ey_nodes
            y component of the electric field
        centers
            coordinates of the nodes
        """
        if self.phi is None or self.phi.size == 0:
            raise RuntimeError("Call solve() first.")
        N = self.N
        Ex_acc = np.zeros(N)
        Ey_acc = np.zeros(N)
        count = np.zeros(N)

        for tri in self.tris:
            p = self.nodes[tri]
            A, grads = self._triangle_area_and_grads(p)
            phi_loc = self.phi[tri]
            E_elem = - (phi_loc[:, None] * grads).sum(axis=0)
            for n in tri:
                Ex_acc[n] += E_elem[0]
                Ey_acc[n] += E_elem[1]
                count[n] += 1.0

        Ex_nodes = Ex_acc / np.maximum(count, 1.0)
        Ey_nodes = Ey_acc / np.maximum(count, 1.0)
        centers = self.nodes
        return Ex_nodes, Ey_nodes, centers
