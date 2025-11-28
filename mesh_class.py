"""
Class for the setup of 2D FEM mesh for electromagnetic field calculations.

Usage:
    from mesh_class import *

Author:
    Florian Meiners - November 5, 2025
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

@dataclass
class Mesh2DRect:
    """
    Class representing the 2D FEM mesh based on equilateral, right-angle triangles.

    Attributes:
    -----------
    xmin: float
        x coordinate of the lower left corner of the mesh
    xmax: float
        x coordinate of the upper right corner of the mesh
    ymin: float
        y coordinate of the lower left corner of the mesh
    ymax: float
        y coordinate of the upper right corner of the mesh
    nx: int
        number of steps in the x direction
    ny: int
        number of steps in the y direction

    Methods:
    -----------
    build(self)
        Sets up the FEM mesh.
    boundary_nodes(self, nodes: np.ndarray, tol: float = 1e-12)
        Declares what nodes belong to the boundary of the spatial domain.
    """
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    nx: int
    ny: int

    # def function(argument) -> expected_return_type:
    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for building the FEM mesh.

        Arguments:
        ----------
        self

        Returns:
        -----------
        nodes: np.ndarray
            Nodes of the FEM mesh.
        tris: np.ndarray
            Triangles of the FEM mesh.
        """
        xs = np.linspace(self.xmin, self.xmax, self.nx + 1)
        ys = np.linspace(self.ymin, self.ymax, self.ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        def idx(i, j):
            return j * (self.nx + 1) + i

        tris = []
        for j in range(self.ny):
            for i in range(self.nx):
                n00 = idx(i, j)
                n10 = idx(i + 1, j)
                n01 = idx(i, j + 1)
                n11 = idx(i + 1, j + 1)
                tris.append([n00, n10, n11])
                tris.append([n00, n11, n01])
        tris = np.asarray(tris, dtype=int)
        return nodes, tris

    def boundary_nodes(self, nodes: np.ndarray, tol: float = 1e-12) -> Dict[str, np.ndarray]:
        """
        Method for declaring the boundary nodes of the FEM mesh.

        Arguments:
        ----------
        self
        nodes: np.ndarray
            Nodes of the FEM mesh.
        tol: float
            Distance tolerance of for nodes to be considered boundary nodes.

        Returns:
        -----------
        Dict of boundary nodes sorted by "left", "right", "top", "bottom".
        """
        x, y = nodes[:, 0], nodes[:, 1]
        left  = np.where(np.isclose(x, self.xmin, atol=tol))[0]
        right = np.where(np.isclose(x, self.xmax, atol=tol))[0]
        bottom= np.where(np.isclose(y, self.ymin, atol=tol))[0]
        top   = np.where(np.isclose(y, self.ymax, atol=tol))[0]
        return {"left": left, "right": right, "bottom": bottom, "top": top}
