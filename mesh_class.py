"""
Class for the setup of 2D FEM mesh for electromagnetic field calculations.

Usage:
    from mesh_class import *

Author:
    Florian Meiners - November 5, 2025; Last updated December 15, 2025
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
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
    Class representing the 2D FEM mesh based on equilateral, right-angle triangles. This class also features a
    representation of triangles via edges rather than vertices, which facilitates the formulation of Nédélec elements.

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
    edges: (n_edges, 2) int
        Edges as pairs of node indices, stored with global orientation (min(node_i, node_j), max(node_i, node_j)).
    tri_edges: (n_tris, 3) int
        For each triangle, indices into `edges` for its three edges.
    tri_edge_signs: (n_tris, 3) int
        For each triangle, the sign (+1/-1) of the local edge orientation relative to the global orientation in `edges`.

    Methods:
    -----------
    build(self)
        Sets up the FEM mesh.
    _build_edges(self) (static method)
         builds the global edge list and list of orientations from a list of triangles
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

        Sets:
        -----------
        edges, tri_edges, tri_edge_signs
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

        edges, tri_edges, tri_edge_signs = self._build_edges(tris)
        self.nodes = nodes
        self.tris = tris
        self.edges = edges
        self.tri_edges = tri_edges
        self.tri_edge_signs = tri_edge_signs

        return nodes, tris

    @staticmethod
    def _build_edges(tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        From a triangle list, builds the unique global edge list and triangle-edge connectivity with orientation signs.

        Parameters:
        -----------
        tris: (n_tris, 3) int

        Returns:
        -----------
        edges: (n_edges, 2) int
            Global edges with canonical orientation (min, max) - independently of what triangle has that edge.
        tri_edges: (n_tris, 3) int
            tri_edges[t, l] = global edge index of local edge l on triangle t (i.e., which one of the edges is edge 0, 1,
             or 2 of a triangle).
        tri_edge_signs: (n_tris, 3) int
            tri_edge_signs[t, l] = +1 if triangle's local orientation of edge l matches the global orientation,
            -1 otherwise (this tells us whether a sign needs to be flipped during FEM assembly).
        """
        n_tris = tris.shape[0]
        edge_map: Dict[Tuple[int, int], int] = {}
        edges = []

        tri_edges = np.empty((n_tris, 3), dtype=int)
        tri_edge_signs = np.empty((n_tris, 3), dtype=int)

        for t, (n0, n1, n2) in enumerate(tris):  # iterates over the triangles with vertices 0, 1, and 2
            local_edges = [(n0, n1), (n1, n2), (n2, n0)]  # defines the directions of local edges (from 0 to 1 and so on)
            for l, (a, b) in enumerate(local_edges):
                # global representation (makes sure that edges are geometrically in the same spot despite orientation
                # mismatches):
                key = (min(a, b), max(a, b))
                if key not in edge_map:
                    edge_id = len(edges)
                    edge_map[key] = edge_id
                    edges.append(key)
                else:
                    edge_id = edge_map[key]

                tri_edges[t, l] = edge_id  # For triangle t and local edge l, record which global edge this is.
                # For triangle t and local edge l, record which global edge this is:
                tri_edge_signs[t, l] = +1 if (a, b) == key else -1

        edges = np.asarray(edges, dtype=int)
        return edges, tri_edges, tri_edge_signs

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

    def boundary_edges(self, tol: float = 1e-12) -> Dict[str, np.ndarray]:
        """
        Return boundary edges grouped by side of the rectangle.

        An edge is on a given side if both its nodes lie on that side.

        Parameters:
        -----------
        self
        tol: float
            Distance tolerance for classification.

        Returns:
        -----------
        dict with keys "left", "right", "bottom", "top" and values arrays of edge indices.
        """
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        edge_x = x[self.edges]
        edge_y = y[self.edges]

        def on_left():
            mask = np.all(np.isclose(edge_x, self.xmin, atol=tol), axis=1)
            return np.where(mask)[0]

        def on_right():
            mask = np.all(np.isclose(edge_x, self.xmax, atol=tol), axis=1)
            return np.where(mask)[0]

        def on_bottom():
            mask = np.all(np.isclose(edge_y, self.ymin, atol=tol), axis=1)
            return np.where(mask)[0]

        def on_top():
            mask = np.all(np.isclose(edge_y, self.ymax, atol=tol), axis=1)
            return np.where(mask)[0]

        return {
            "left": on_left(),
            "right": on_right(),
            "bottom": on_bottom(),
            "top": on_top(),
        }
