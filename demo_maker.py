"""
Collection of functions to set up electro- and magneto(quasi)static FEM problems from the GUI

Usage:
    import demo_maker

Author:
    Florian Meiners - January 20, 2026

Functions:
-----------
make_two_material_demo(mesh: Mesh2DRect, x_split: float = 0.6,
                       eps_left: float = 1.0, eps_right: float = 5.0, charge_function=None, mat_distribution=None):
    creates a 2D demo of the FEM calculation with a specified material and charge distribution across a specified domain

make_two_material_demo_magnetic_nedelec(mesh: Mesh2DRect, vector_source, mat_distribution=None, x_split: float = 0.5,
                                        eps_left: float = 10.0, eps_right: float = 1.0, current_function=None,
                                        boundary_type="Dirichlet"):
    Creates a 2D demo of the magnetic FEM calculation with specified material and current distributions across a
    specified domain; the implementation uses a Nédélec formulation of this problem with in-plane current density J(x,y).
    This function now allows the user to choose between Dirichlet and Neumann boundary conditions.

make_two_material_demo_magnetic(mesh: Mesh2DRect, vector_source, mat_distribution, x_split: float = 0.5,
                                eps_left: float = 10.0, eps_right: float = 1.0, current_function=None):
    creates a 2D demo of the magnetic FEM calculation with specified material and current distributions across a
    specified domain; unlike the Nédélec formulation, this is a magnetoquasistatic extension of the electrostatic case
    and therefore uses node-based formulations.

make_two_material_demo_magnetostatic(mesh: Mesh2DRect, mat_distribution, M_callable, x_split: float = 0.5,
                                     eps_left: float = 10.0, eps_right: float = 1.0)
    creates a 2D demo of the magnetostatic FEM calculation with specified material and current distributions across a
    specified domain; the sources are permanent magnets in this case
"""

from electrostatics_class import *
from magnetic_planar_class import *
from magnetostatic_planar_class import *
from mesh_class import *

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def make_two_material_demo(mesh: Mesh2DRect, x_split: float = 0.6, eps_left: float = 1.0, eps_right: float = 5.0,
                           charge_function=None, mat_distribution=None):
    nodes, tris = mesh.build()
    mat = MaterialEpsilon(mat_distribution if mat_distribution is not None
                          else (lambda x, y: eps_left if x <= x_split else eps_right))
    src = ChargeDensity(charge_function if charge_function is not None else (lambda x, y: 0.0))
    bnds = mesh.boundary_nodes(nodes)
    left_bc = DirichletBC(bnds["left"], lambda x, y: 0.0)
    right_bc = DirichletBC(bnds["right"], lambda x, y: 0.0)
    top_bc = DirichletBC(bnds["top"], lambda x, y: 0.0)
    bottom_bc = DirichletBC(bnds["bottom"], lambda x, y: 0.0)

    prob = Electrostatics2D(mesh, mat, src)
    prob.assemble_by_elements()
    prob.apply_dirichlet([left_bc, right_bc, top_bc, bottom_bc])
    phi = prob.solve()
    Ex, Ey, centers = prob.electric_field()
    return prob, phi, Ex, Ey, centers, nodes, tris


def make_two_material_demo_magnetic_nedelec(mesh: Mesh2DRect, vector_source, mat_distribution=None, x_split: float = 0.5,
                                            eps_left: float = 10.0, eps_right: float = 1.0,
                                            current_function=None, boundary_type="Dirichlet"):
    nodes, tris = mesh.build()
    mat = MaterialMu(mat_distribution if mat_distribution is not None
                     else (lambda x, y: eps_left if x <= x_split else eps_right))
    src = CurrentDensity(current_function if current_function is not None else (lambda x, y: 0.0))
    prob = Magnetic2DHcurl(mesh, mat, src, vector_source=vector_source)
    prob.assemble()
    bnd_edges = mesh.boundary_edges()
    all_bnd_edges = np.concatenate([bnd_edges["left"], bnd_edges["right"], bnd_edges["top"], bnd_edges["bottom"]])

    match boundary_type:
        case "Dirichlet":
            bc_all = DirichletBCMagneticEdge(edges=all_bnd_edges, value=(lambda x, y: 0.0))
            prob.apply_dirichlet([bc_all])
        case "Neumann":
            bc_all = NeumannBCMagneticEdge(edges=all_bnd_edges, value=(lambda x, y: (0.0, 5.0)))
            prob.apply_neumann([bc_all])
    A_v = prob.solve()
    Bz, centers = prob.magnetic_field()
    return prob, A_v, Bz, centers, nodes, tris


def make_two_material_demo_magnetic(mesh: Mesh2DRect, mat_distribution, x_split: float = 0.5,
                                    eps_left: float = 10.0, eps_right: float = 1.0,
                                    current_function=None):
    nodes, tris = mesh.build()
    mat = MaterialMu(mat_distribution if mat_distribution is not None
                     else (lambda x, y: eps_left if x <= x_split else eps_right))
    src = CurrentDensity(current_function if current_function is not None else (lambda x, y: 0.0))
    bnds = mesh.boundary_nodes(nodes)
    left_bc = DirichletBCMagnetic(bnds["left"], lambda x, y: 0.0)
    right_bc = DirichletBCMagnetic(bnds["right"], lambda x, y: 0.0)
    top_bc = DirichletBCMagnetic(bnds["top"], lambda x, y: 0.0)
    bottom_bc = DirichletBCMagnetic(bnds["bottom"], lambda x, y: 0.0)

    prob = Magnetic2D(nodes, tris, mat, src)
    prob.assemble()
    prob.apply_dirichlet([left_bc, right_bc, top_bc, bottom_bc])
    A_v = prob.solve()
    Bx, By, centers = prob.magnetic_field()
    return prob, A_v, Bx, By, centers, nodes, tris


def make_two_material_demo_magnetostatic(mesh: Mesh2DRect, mat_distribution, M_callable, x_split: float = 0.5,
                                        eps_left: float = 10.0, eps_right: float = 1.0):
    nodes, tris = mesh.build()

    mat = MaterialMu(mat_distribution if mat_distribution is not None
                     else (lambda x, y: eps_left if x <= x_split else eps_right))
    problem = Magnetostatic2D(nodes, tris, material=mat, remanence=RemanentFluxDensityInPlane(Br=M_callable))
    problem.assemble()
    bnds = mesh.boundary_nodes(nodes)
    left_bc = DirichletBCMagnetic(bnds["left"], lambda x, y: 0.0)
    right_bc = DirichletBCMagnetic(bnds["right"], lambda x, y: 0.0)
    top_bc = DirichletBCMagnetic(bnds["top"], lambda x, y: 0.0)
    bottom_bc = DirichletBCMagnetic(bnds["bottom"], lambda x, y: 0.0)
    problem.apply_dirichlet([left_bc, right_bc, top_bc, bottom_bc])

    A_v = problem.solve()
    Bx, By, centers = problem.magnetic_field()

    return problem, A_v, Bx, By, centers, nodes, tris
