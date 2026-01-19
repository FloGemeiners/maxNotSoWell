"""
Classes formalizing the definition of finite elements as subclasses of the abstract superclass FiniteElement.

Usage:
    from finite_element_classes import *

Author:
    Florian Meiners - December 16, 2025
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class FiniteElement(ABC):
    """
    Abstract base class for all finite elements on a given reference cell.

    This class is dimension- and space-agnostic; concrete subclasses implement the actual reference basis and
    (if needed) derivative information.
    """

    # functional properties of the element and basis
    @property
    @abstractmethod
    def space_type(self) -> str:
        """Function space this element is conforming to, e.g. 'H1', 'H(curl)'."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Spatial dimension of the reference cell (2 for 2D elements like triangles)."""
        ...

    @property
    @abstractmethod
    def value_shape(self) -> Tuple[int, ...]:
        """Shape of the FE function values: () for scalar, (dim, ) for vector."""
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Polynomial order of the element basis functions."""
        ...

    @property
    @abstractmethod
    def reference_cell(self) -> str:
        """Type of reference cell, e.g. 'triangle'."""
        ...

    # layout of the degrees of freedom
    @abstractmethod
    def num_local_dofs(self) -> int:
        """Total number of local DOFs per cell (either living on edges or vertices of the element)."""
        ...

    @abstractmethod
    def num_entity_dofs(self, entity_dim: int) -> int:
        """
        Number of DOFs per topological entity of given dimension.

        entity_dim = 0 : vertex,
                     1 : edge,
                     2 : face,
                     3 : cell interior.
        """
        ...

    # functional evaluation on the reference element
    @abstractmethod
    def evaluate_reference_basis(self, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Evaluate basis functions on the reference cell.

        Parameters:
        -----------
        ref_q_points: (n_q, dim) array
            Quadrature points in reference coordinates.

        Returns:
        -----------
        values: array
            For scalar elements (H1): shape (n_q, n_dofs). For vector elements (H(curl)): shape (n_q, n_dofs, dim).
        """
        ...

    def evaluate_reference_gradients(self, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Evaluate gradients on the reference cell
        (H1 elements only, this does not make sense for vector-valued basis functions).

        Returns:
        -----------
        grads: (n_q, n_dofs, dim)
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement gradients")

    def evaluate_reference_curl(self, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Evaluate scalar curl of vector basis on the reference cell (2D H(curl) elements).

        Returns:
        -----------
        curl: (n_q, n_dofs)
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement curl")

    # helpers for the mapping of the reference element to physical elements (right now triangles in 2D only)
    @staticmethod
    def _compute_affine_jacobian(cell_vertices: np.ndarray):
        """
        Compute affine mapping Jacobian for a linear triangle in 2D.

        Reference triangle: (0,0), (1,0), (0,1)
        Physical vertices: cell_vertices[0], [1], [2]

        Returns:
        -----------
        J: (2,2) array
        detJ: float
        invJT: (2,2) array
        x0: (2, ) array, image of reference origin
        """
        if cell_vertices.shape != (3, 2):
            raise ValueError("cell_vertices must have shape (3, 2) for 2D triangles")
        x0, x1, x2 = cell_vertices
        J = np.column_stack((x1 - x0, x2 - x0))
        detJ = float(np.linalg.det(J))
        if detJ == 0.0:
            raise ValueError("Degenerate cell with zero Jacobian determinant")
        invJT = np.linalg.inv(J).T
        return J, detJ, invJT, x0

    # evaluation of basis functions and derivative information on the physical element
    def evaluate_basis(self, cell_vertices: np.ndarray, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Evaluate basis functions on a physical cell.

        Default implementation:
        - For H1: same as reference basis (values are invariant under affine mapping).
        - For H(curl): subclasses should override to apply covariant Piola map.
        """
        if self.space_type == "H1":
            return self.evaluate_reference_basis(ref_q_points)
        raise NotImplementedError(f"{self.__class__.__name__} must override evaluate_basis" 
                                  f"for space_type='{self.space_type}'")

    def evaluate_gradients(self, cell_vertices: np.ndarray, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Evaluate gradients on a physical cell
        (H1 scalar elements only, this does not make sense for vector-valued basis functions).

        grad_x phi = J^{-T} grad_hat phi

        Parameters:
        -----------
        cell_vertices : (3,2)
        ref_q_points : (n_q,2)

        Returns:
        -----------
        grads : (n_q, n_dofs, dim)
        """
        if self.space_type != "H1":
            raise NotImplementedError("evaluate_gradients only defined for H1 elements")

        ref_grads = self.evaluate_reference_gradients(ref_q_points)
        _, _, invJT, _ = self._compute_affine_jacobian(cell_vertices)
        return np.einsum("ij,qkj->qki", invJT, ref_grads)

    def evaluate_curl(self, cell_vertices: np.ndarray, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Evaluate scalar curl of H(curl) basis on a physical cell in 2D.

        curl_x N = (1/detJ) * curl_hat N
        """
        if self.space_type != "H(curl)":
            raise NotImplementedError("evaluate_curl only defined for H(curl) elements")

        ref_curl = self.evaluate_reference_curl(ref_q_points)
        _, detJ, _, _ = self._compute_affine_jacobian(cell_vertices)
        return ref_curl / detJ


class LagrangeP1Tri(FiniteElement):
    """
    Linear Lagrange element on a 2D reference triangle (H1-conforming, scalar).
    """

    @property
    def space_type(self) -> str:
        return "H1"

    @property
    def dim(self) -> int:
        return 2

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return ()

    @property
    def order(self) -> int:
        return 1

    @property
    def reference_cell(self) -> str:
        return "triangle"

    def num_local_dofs(self) -> int:
        return 3

    def num_entity_dofs(self, entity_dim: int) -> int:
        # 1 DOF per vertex, none on edges/faces
        if entity_dim == 0:
            return 1
        if entity_dim in (1, 2, 3):
            return 0
        raise ValueError("entity_dim must be 0,1,2,3")

    # evaluation on the basis element

    def evaluate_reference_basis(self, ref_q_points: np.ndarray) -> np.ndarray:
        """
        P1 basis on reference triangle with vertices (0,0), (1,0), (0,1).

        phi0 = 1 - xi - eta
        phi1 = xi
        phi2 = eta
        """
        xi = ref_q_points[:, 0]
        eta = ref_q_points[:, 1]
        n_q = ref_q_points.shape[0]

        vals = np.empty((n_q, 3), dtype=float)
        vals[:, 0] = 1.0 - xi - eta
        vals[:, 1] = xi
        vals[:, 2] = eta
        return vals

    def evaluate_reference_gradients(self, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Gradients of P1 basis are constant on the reference element.
        """
        n_q = ref_q_points.shape[0]
        grads = np.empty((n_q, 3, 2), dtype=float)

        # grad phi0 = (-1, -1)
        grads[:, 0, 0] = -1.0
        grads[:, 0, 1] = -1.0

        # grad phi1 = (1, 0)
        grads[:, 1, 0] = 1.0
        grads[:, 1, 1] = 0.0

        # grad phi2 = (0, 1)
        grads[:, 2, 0] = 0.0
        grads[:, 2, 1] = 1.0

        return grads


class NedelecFirstKindTriP1(FiniteElement):
    """
    Lowest-order Nédélec (first family) element on a 2D triangle (H(curl)-conforming).

    - 1 DOF per edge (tangential line integral).
    It is crucial that the degrees of freedom of such an element live on the element faces rather than the vertices.
    Rather than enforcing continuity of the solution through shared vertices (which happens in the Lagrange elements
    above), vector-valued basis functions are designed to be continuous at the edges.
    - 3 local DOFs per triangle.
    - Vector-valued basis functions.
    """

    @property
    def space_type(self) -> str:
        return "H(curl)"

    @property
    def dim(self) -> int:
        return 2

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return (2,)  # 2D vector field

    @property
    def order(self) -> int:
        return 1

    @property
    def reference_cell(self) -> str:
        return "triangle"

    def num_local_dofs(self) -> int:
        return 3  # 3 edges

    def num_entity_dofs(self, entity_dim: int) -> int:
        # 1 DOF per edge, none on vertices / interior
        if entity_dim == 1:
            return 1
        if entity_dim in (0, 2, 3):
            return 0
        raise ValueError("entity_dim must be 0,1,2,3")

    # evaluation on the basis element
    def evaluate_reference_basis(self, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Nédélec basis on reference triangle with vertices (0,0), (1,0), (0,1).

        Using the standard definition of the curl in two dimensions in terms of barycentric coordinates lambda_i:

            N0 = lambda1 * grad(lambda2) - lambda2 * grad(lambda1)
            N1 = lambda2 * grad(lambda3) - lambda3 * grad(lambda2)
            N2 = lambda3 * grad(lambda1) - lambda1 * grad(lambda3)

        which simplifies to:

            N0(xi,eta) = (1 - eta,     xi  )
            N1(xi,eta) = ( -eta  ,     xi  )
            N2(xi,eta) = ( -eta  ,   xi - 1)
        """
        xi = ref_q_points[:, 0]
        eta = ref_q_points[:, 1]
        n_q = ref_q_points.shape[0]

        vals = np.empty((n_q, 3, 2), dtype=float)

        # edge zero
        vals[:, 0, 0] = 1.0 - eta
        vals[:, 0, 1] = xi

        # edge one
        vals[:, 1, 0] = -eta
        vals[:, 1, 1] = xi

        # edge two
        vals[:, 2, 0] = -eta
        vals[:, 2, 1] = xi - 1.0

        return vals

    def evaluate_reference_curl(self, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Scalar 2D curl of each reference basis function.

        curl(N) = d(N_y)/d xi - d(N_x)/d eta = 2  (constant on the reference element).
        """
        n_q = ref_q_points.shape[0]
        return 2.0 * np.ones((n_q, 3), dtype=float)

    # mapping to the physical element (H(curl) Piola)
    def evaluate_basis(self, cell_vertices: np.ndarray, ref_q_points: np.ndarray) -> np.ndarray:
        """
        Evaluate Nédélec basis on a physical triangle (2D, H(curl)).

        Uses the covariant Piola transform:
            N(x) = J^{-T} * N_hat(x_hat)
        """
        ref_vals = self.evaluate_reference_basis(ref_q_points)
        _, _, invJT, _ = self._compute_affine_jacobian(cell_vertices)

        # note to myself... Use the Einstein summation convention to apply J^{-T} (inverse transposed Jacobian) to every
        # basis vector at every quadrature point (which means left-multiplying by J^{-T}):
        # invJT: (2,2), ref_vals: (n_q,3,2) -> out: (n_q,3,2)
        vals = np.einsum("ij,qkj->qki", invJT, ref_vals)
        return vals
