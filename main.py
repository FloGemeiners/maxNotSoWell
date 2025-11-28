"""
Main script to run the electrostatic FEM field calculation with.

Usage:
    python3 main.py or hit the play button

Author:
    Florian Meiners - November 5, 2025; Last updated November 27, 2025

Functions:
-----------
make_two_material_demo(mesh: Mesh2DRect, x_split: float = 0.6,
                       eps_left: float = 1.0, eps_right: float = 5.0, charge_function=None, mat_distribution=None):
    creates a 2D demo of the FEM calculation with a specified material and charge distribution across a specified domain

plot_electric_potential_and_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh: bool = True):
    plots the potential and the electric field for a given solution

make_two_material_demo_magnetic(mesh: Mesh2DRect, x_split: float = 0.5, eps_left: float = 10.0, eps_right: float = 1.0,
                                current_function=None, mat_distribution=None):
    creates a 2D demo of the magnetic FEM calculation with a specified material and current distribution across a
    specified domain

plot_magnetic_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh: bool = True)
    plots the magnetic field over contours of the z component of the magnetic vector potential
"""
from pydoc_data.topics import topics

import matplotlib.pyplot as plt
import demo_functions
from electrostatics_class import *
from magnetic_planar_class import *
from mesh_class import *

plt.rcParams.update({'text.usetex':True,
                     'font.family':'Helvetica'})

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

    prob = Electrostatics2D(nodes, tris, mat, src)
    prob.assemble()
    prob.apply_dirichlet([left_bc, right_bc, top_bc, bottom_bc])
    phi = prob.solve()
    Ex, Ey, centers = prob.electric_field()
    return prob, phi, Ex, Ey, centers, nodes, tris


def make_two_material_demo_magnetic(mesh: Mesh2DRect, x_split: float = 0.5,
                                    eps_left: float = 10.0, eps_right: float = 1.0,
                                    current_function=None, mat_distribution=None):
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


def plot_electric_potential_and_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh: bool = True):
    import matplotlib.tri as mtri
    triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig1 = plt.figure(figsize=(7, 3.0))
    ax1 = fig1.add_subplot(111)
    tcf = ax1.tricontourf(triobj, phi, levels=30, cmap='cubehelix')
    plt.colorbar(tcf, ax=ax1, label=r"Potential $\varphi$")
    ax1.set_aspect('equal')
    ax1.set_title("Electrostatics (P1 FEM): Potential")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    fig2 = plt.figure(figsize=(7, 3.0))
    ax2 = fig2.add_subplot(111)
    ax2.tricontour(triobj, phi, levels=20, linewidths=0.8)
    step = max(1, len(nodes)//3000)
    idx = np.where((np.sqrt(Ex[::step] ** 2 + Ey[::step] ** 2)) != 0)
    ax2.quiver(nodes[::step, 0][idx], nodes[::step, 1][idx],
               Ex[::step][idx] / (np.sqrt(Ex[::step] ** 2 + Ey[::step] ** 2))[idx],
               Ey[::step][idx] / (np.sqrt(Ex[::step] ** 2 + Ey[::step] ** 2))[idx],
               [1 / (np.sqrt(Ex[::step] ** 2 + Ey[::step] ** 2))[idx]],
               angles='xy', scale_units='xy', scale=50, cmap='cubehelix')
    # ax2.quiver(nodes[::step, 0], nodes[::step, 1], Ex[::step], Ey[::step])
    ax2.set_aspect('equal')
    ax2.set_title(r"Electric field $\mathbf{E}$ over contours of the potential $\varphi$")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")

    if show_mesh:
        ax1.triplot(triobj, linewidth=0.4)
        # ax2.triplot(triobj, linewidth=0.4)  # plot triangulation for the electric field as well

    plt.tight_layout()
    if outpath_png is not None:
        fig1.savefig(outpath_png.replace(".png", "_potential.png"), dpi=160, bbox_inches="tight")
        fig2.savefig(outpath_png.replace(".png", "_field.png"), dpi=160, bbox_inches="tight")
    else:
        plt.show()
    return fig1, fig2


def plot_magnetic_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh: bool = True):
    import matplotlib.tri as mtri
    triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)

    fig = plt.figure(figsize=(7, 3.0))
    ax = fig.add_subplot(111)
    ax.tricontour(triobj, phi, levels=20, linewidths=0.8)
    step = max(1, len(nodes)//3000)
    ax.quiver(nodes[::step, 0], nodes[::step, 1], Ex[::step], Ey[::step])
    ax.set_aspect('equal')
    ax.set_title(r"Magnetic field $\mathbf{B}$ over contours of the "
                 r"$z$-component of the magnetic vector potential $\mathbf{A}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    if show_mesh:
        ax.triplot(triobj, linewidth=0.4, alpha=0.4, color="black")

    plt.tight_layout()
    if outpath_png is not None:
        fig.savefig(outpath_png.replace(".png", "_field.png"), dpi=160, bbox_inches="tight")
    else:
        plt.show()
    return fig


electric = False

if __name__ == "__main__":
    mesh_obj = Mesh2DRect(0.0, 1.0, 0.0, 1.0, nx=91, ny=91)
    if electric:
        _, phi, Ex, Ey, _, nodes, tris = make_two_material_demo(mesh_obj,
                                                                charge_function=demo_functions.rho_gauss_2(),
                                                                mat_distribution=demo_functions.mat_inhomogeneous_2())
        plot_electric_potential_and_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh=True)

    else:
        _, A_z, Bx, By, _, nodes, tris = make_two_material_demo_magnetic(mesh_obj,
                                                                         mat_distribution=(lambda x, y: 1.0),
                                                                         current_function=demo_functions.line_conductor())
        plot_magnetic_field(nodes, tris, A_z, Bx, By, outpath_png=None, show_mesh=True)
    print("Demo done.")
