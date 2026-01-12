"""
Former main script to run the electrostatic FEM field calculation with.

Usage:
    python3 main_old.py or hit the play button

Author:
    Florian Meiners - November 5, 2025; Last updated January 12, 2026

Functions:
-----------
make_two_material_demo(mesh: Mesh2DRect, x_split: float = 0.6,
                       eps_left: float = 1.0, eps_right: float = 5.0, charge_function=None, mat_distribution=None):
    creates a 2D demo of the FEM calculation with a specified material and charge distribution across a specified domain

plot_electric_potential_and_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh: bool = True):
    plots the potential and the electric field for a given solution

make_two_material_demo_magnetic(mesh: Mesh2DRect, vector_source, mat_distribution, x_split: float = 0.5,
                                eps_left: float = 10.0, eps_right: float = 1.0, current_function=None):
    creates a 2D demo of the magnetic FEM calculation with specified material and current distributions across a
    specified domain; as of January 4, 2026, the implementation is extended by a Nédélec formulation of this problem
    with in-plane current density J(x,y)

make_two_material_demo_magnetostatic(mesh: Mesh2DRect, mat_distribution, M_callable, x_split: float = 0.5,
                                        eps_left: float = 10.0, eps_right: float = 1.0)
    creates a 2D demo of the magnetostatic FEM calculation with specified material and current distributions across a
    specified domain; the sources are permanent magnets in this case

plot_magnetic_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh: bool = True, title: str = None):
    triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    plots the magnetic field over contours of the z component of the magnetic vector potential

plot_magnetic_flux_density_heatmap(nodes: np.ndarray, tris: np.ndarray, Bz_cells: np.ndarray,
                                   outpath_png: str | None = None, show_mesh: bool = True):
    plots the z-component of the magnetic field; since the field only points into and out of the plane, there is no use
    in arrows in this scenario
"""

import matplotlib.pyplot as plt
import demo_functions
from electrostatics_class import *
from magnetic_planar_class import *
from magnetostatic_planar_class import *
from mesh_class import *
import matplotlib.tri as mtri

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

    prob = Electrostatics2D(mesh, mat, src)
    # prob.assemble()
    prob.assemble_by_elements()
    prob.apply_dirichlet([left_bc, right_bc, top_bc, bottom_bc])
    phi = prob.solve()
    Ex, Ey, centers = prob.electric_field()
    return prob, phi, Ex, Ey, centers, nodes, tris


def make_two_material_demo_magnetic(mesh: Mesh2DRect, vector_source, mat_distribution, x_split: float = 0.5,
                                    eps_left: float = 10.0, eps_right: float = 1.0,
                                    current_function=None):
    nodes, tris = mesh.build()
    mat = MaterialMu(mat_distribution if mat_distribution is not None
                     else (lambda x, y: eps_left if x <= x_split else eps_right))
    src = CurrentDensity(current_function if current_function is not None else (lambda x, y: 0.0))
    # bnds = mesh.boundary_nodes(nodes)
    # left_bc = DirichletBCMagnetic(bnds["left"], lambda x, y: 0.0)
    # right_bc = DirichletBCMagnetic(bnds["right"], lambda x, y: 0.0)
    # top_bc = DirichletBCMagnetic(bnds["top"], lambda x, y: 0.0)
    # bottom_bc = DirichletBCMagnetic(bnds["bottom"], lambda x, y: 0.0)

    # prob = Magnetic2D(nodes, tris, mat, src)
    prob = Magnetic2DHcurl(mesh, mat, src, vector_source=vector_source)
    prob.assemble()
    # prob.apply_dirichlet([left_bc, right_bc, top_bc, bottom_bc])
    bnd_edges = mesh.boundary_edges()
    all_bnd_edges = np.concatenate([bnd_edges["left"], bnd_edges["right"], bnd_edges["top"], bnd_edges["bottom"]])
    bc_outer = DirichletBCMagneticEdge(edges=all_bnd_edges)  # value=None ⇒ homogeneous
    prob.apply_dirichlet([bc_outer])
    A_v = prob.solve()
    # Bx, By, centers = prob.magnetic_field()
    Bz, centers = prob.magnetic_field()
    return prob, A_v, Bz, centers, nodes, tris


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


def plot_magnetic_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh: bool = True, title: str = None):
    triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)

    fig = plt.figure(figsize=(7, 3.0))
    ax = fig.add_subplot(111)
    ax.tricontour(triobj, phi, levels=20, linewidths=0.8)
    step = max(1, len(nodes)//3000)
    ax.quiver(nodes[::step, 0], nodes[::step, 1], Ex[::step], Ey[::step])
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Magnetic Field")
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

def plot_magnetic_flux_density_heatmap(nodes: np.ndarray, tris: np.ndarray, Bz_cells: np.ndarray,
                                       outpath_png: str | None = None, show_mesh: bool = True):
    triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)

    fig = plt.figure(figsize=(7, 3.0))
    ax = fig.add_subplot(111)

    tpc = ax.tripcolor(triobj, Bz_cells, shading="flat")
    cbar = fig.colorbar(tpc, ax=ax)
    cbar.set_label(r"$B_z$")

    ax.set_aspect("equal")
    ax.set_title(r"Out-of-plane magnetic flux density $B_z$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    if show_mesh:
        ax.triplot(triobj, linewidth=0.4, alpha=0.4, color="black")

    plt.tight_layout()

    if outpath_png is not None:
        if outpath_png.endswith(".png"):
            save_path = outpath_png.replace(".png", "_Bz.png")
        else:
            save_path = outpath_png
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    else:
        plt.show()

    return fig

scenario = "Magnetostatic"

if __name__ == "__main_old__":
    mesh_obj = Mesh2DRect(0.0, 1.0, 0.0, 1.0, nx=91, ny=91)
    match scenario:
        case "Electrostatic":
            _, phi, Ex, Ey, _, nodes, tris = make_two_material_demo(mesh_obj,
                                                                    charge_function=demo_functions.rho_gauss_2(),
                                                                    mat_distribution=demo_functions.mat_inhomogeneous_2())
            plot_electric_potential_and_field(nodes, tris, phi, Ex, Ey, outpath_png=None, show_mesh=True)
        case "Magnetoquasistatic":
            J_line = demo_functions.make_line_current(p0=(0.25, 0.5), p1=(0.75, 0.5), J0=1.0, thickness=0.01)
            J_circle = demo_functions.make_circular_current(center=(0.5, 0.5), radius=0.2, J0=-10.0, thickness=0.01)
            _, A_z, Bz, _, nodes, tris = make_two_material_demo_magnetic(mesh_obj, vector_source=J_line,
                                                                         mat_distribution=None, eps_left=1, eps_right=1)
            plot_magnetic_flux_density_heatmap(nodes, tris, Bz)
        case "Magnetostatic":
            Br_bar = demo_functions.make_rectangular_Br(center=(0.5, 0.5), half_sizes=(0.15, 0.07), Br0=1.0,
                                                        direction=(1.0, 0.0), smoothing=0.01)
            Br_horseshoe = demo_functions.make_horseshoe_Br(center=(0.5, 0.5), leg_length=0.35, leg_thickness=0.08,
                                                            gap=0.10, yoke_thickness=0.08, Br0=1.0, angle_rad=0.0,
                                                            smoothing=0.01, opening="up", gap_flux_lr=+1)
            _, A_z, Bx, By, _, nodes, tris = make_two_material_demo_magnetostatic(mesh_obj, mat_distribution=None,
                                                                                  M_callable=Br_bar,
                                                                                  eps_left=1, eps_right=1)
            plot_magnetic_field(nodes, tris, A_z, Bx, By, outpath_png=None, show_mesh=True,
                                title=r"Magnetic field $\mathbf{B}$")

    print("Demo done.")

