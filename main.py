"""
Main script and handler of the graphic user interface of maxNotSoWell. This is used to run the FEM field calculation.

Contains a number of classes for running the GUI so the code does not need to be touched while doing the calculations.

Usage:
    python3 main.py or hit the play button

Author:
    Florian Meiners - November 5, 2025; Last updated January 21, 2026
"""
import sys
from abc import abstractmethod

from PyQt6.QtCore import *
import PyQt6.QtGui as QtGui
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLabel, QLineEdit,
                             QMainWindow, QPushButton, QVBoxLayout, QWidget, QGridLayout, QFileDialog)

import matplotlib
matplotlib.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.tri as mtri
import pickle

import demo_functions
from mesh_class import *
import demo_maker

class MplCanvas(FigureCanvasQTAgg):
    """
    Canvas Class to handle plots of the FEM solutions. To be used in instances of windows
    (as widgets per the PyQT6 documentation).

    Attributes:
    -----------
        axes: axes object
            axes to plot the respective data in
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Constructor.

        Parameters:
        -----------
            parent: QWidget
                parent of the widget (unused right now)
            width: int
                width of the canvas
            height: int
                height of the canvas
            dpi: int
                dots per inch of the figure
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class PotentialWindow(QWidget):
    """
    Extra window, inherits from the QWidget class. This is used (and passed to the MainWindow) to handle the plot of the
    electrostatic potential (if needed).

    Attributes:
    -----------
        potential_plot_canvas: MplCanvas
            canvas to plot the potential in
    """
    def __init__(self):
        """
        Constructor. Calls the constructor of the parent class.

        Modified parameters of the parent class:
        ----------------------------------------
            windowTitle, layout
        """
        super().__init__()
        self.setWindowTitle("Potential")
        self.potential_plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout = QVBoxLayout()
        layout.addWidget(self.potential_plot_canvas)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    """
    Main window inherits from a QMainWindow class, which is used to handle user input and plot the data.

    Attributes:
    -----------
        sc: MplCanvas
            canvas to plot the potential in
        potential_wíndow: PotentialWindow (defaults to None)
            additional window to plot the potential in
        q_combo_scenario_selector: QComboBox
            drop-down menu to select scenario (Electrostatic or Magneto(quasi)static)
        directory_selector_button: QPushButton
            opens file dialog to select directory for streaming pickle file
        checkbox_show_grid: QCheckBox
            checkbox to select whether to plot the triangulation in the FEM solution
        checkbox_plot_potential: QCheckBox
            checkbox to select whether to plot the potential in Electrostatic FEM
        checkbox_in_plane: QCheckBox
            checkbox to select whether the current in magnetoquasistatic FEM is in-plane
        checkbox_show_magnet: QCheckBox
            checkbox to select whether the shape of the magnet should be shown in the magnetostatic case
        q_combo_source_selector: QComboBox
            drop-down menu to select the source term for the FEM calculation
        q_combo_material_selector: QComboBox
            drop-down menu to select material distribution across the spatial domain
        eps_i_line_edit (i = 1, ..., 4): QLineEdit
            input line for the different permittivity values in heterogeneous material
        mu_i_line_edit (i = 1, ..., 4): QLineEdit
            input line for the different permeability values in heterogeneous material
        rho_line_edit: QLineEdit
            input line for the charge density value
        j_line_edit: QLineEdit
            input line for the current density value
        B_line_edit: QLineEdit
            input line for the remanent flux density value
        eps_i_value, mu_i_value (i = 1, ..., 4), rho_value, j_value, B_value: Float
            numerical values of the material and source parameters (see above)
        eps_label, mu_label, rho_label, j_label, B_label: QLabel
            labels for the QLineEdit input lines;
            these are attributes of the class so they can be turned on and off as needed
        potential_checked: Boolean
            shows whether the potential should be plotted or not
        in_plane_checked: Boolean
            shows whether the current is in-plane in the magnetoquasistatic case
        show_grid_checked: Boolean
            shows whether the grid should be plotted or not
        show_magnet_checked: Boolean
            shows whether the permanent magnet should be displayed or not
        selected_scenario: String
            information about the selected scenario
        selected_source: String
            information about the selected source term
        selected_material: String
            information about the selected material distribution
        boundary_type: String
            information about the type of boundary conditions
        line_output: QLineEdit
            line to give information to the user (for example when the input values aren't feasible)
        simulate_new: Boolean
            whether to run the FEM simulation anew
            (automatically set to True whenever a parameter or the scenario is changed)

    Methods:
    -----------
        scenario_changed(self):
            sets up the GUI according to the selected scenario; this means (de-)activating the respective QLineEdit,
            QCheckBox, and QLabel attributes
        source_changed(self):
            sets the information about the seleted source term
        material_changed(self):
            sets the information about the selected material distribution
        boundary_changed(self):
            sets the information about the boundary conditions to applied to the FEM simulation
        start_fem(self):
            starts the FEM calculation
        eps_i_changed, mu_i_changed (i = 1, ..., 4), rho_changed, j_changed, B_changed:
            set the values of the parameters as per the respective QLineInputs
        checkbox_show_grid_checked(self):
            handles the decisions of whether to plot the triangulation
        checkbox_plot_potential(self):
            handles the decisions of whether to plot the potential in Electrostatic FEM
        checkbox_in_plane(self):
            decides whether the current is in-plane in the magnetoquasistatic case
        checkbox_show_magnet_checked(self, s):
            decides whether the shape of the permanent magnet should be shown in the magnetostatic case
        on_click_line_edit(self):
            runs the simulation at the press of the enter key
        call_directory_selector(self):
            opens a file selection dialog for the selection of the location of pickled simulation data
        plot_electric_potential_and_field(self, nodes, tris, phi, Ex, Ey, show_mesh, plot_potential):
                                        triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
            plots the electric field and potential (if desired); the electric field is displayed in the main window,
            while an additional window is shown for the potential if need be
        plot_magnetic_field(self, nodes, tris, A_v, Bx, By, show_mesh):
            plots the magnetic field
        plot_magnetic_flux_density_heatmap(self, nodes, tris, Bz_cells, show_mesh):
            plots the out-of-plane component of the magnetic field
        plot_magnet_shape(self, center, half_sizes, angle_rad, ):
            plots the shape of the permanent magnet in the magnetostatic case
        closeEvent(self, event):
            closes all windows when the program is terminated by closing the main window
    """
    def __init__(self):
        """
        Constructor. Calls the constructor of the parent class. Most functionality is realized in the methods below,
        but the constructor is where the layout is set.

        Modified parameters of the parent class:
        ----------------------------------------
            windowTitle, layout
        """
        super().__init__()

        # LAYOUT:
        self.setWindowTitle("maxNotSoWell")  # for ultimate adjustability
        self.potential_window = None
        dummy_placeholder_h = QLabel("")
        dummy_placeholder_h.setFixedWidth(150)
        dummy_placeholder_v = QLabel("")
        dummy_placeholder_v.setFixedHeight(30)

        # Drop-down menu for scenario selection
        self.q_combo_scenario_selector = QComboBox()
        self.q_combo_scenario_selector.addItems(["Electrostatic", "Magnetostatic", "Magnetoquasistatic"])
        self.q_combo_scenario_selector.setFixedWidth(200)

        # checkboxes for general options
        self.checkbox_show_grid = QCheckBox("Show grid")
        self.checkbox_show_grid.setFixedHeight(30)
        self.checkbox_plot_potential = QCheckBox("Plot potential")
        self.checkbox_plot_potential.setFixedHeight(30)
        self.checkbox_in_plane = QCheckBox("In-plane current")
        self.checkbox_in_plane.setFixedHeight(30)
        self.checkbox_in_plane.setEnabled(False)
        self.checkbox_show_magnet = QCheckBox("Show magnet")
        self.checkbox_show_magnet.setFixedHeight(30)
        self.checkbox_show_magnet.setEnabled(False)
        self.q_combo_boundary_selector = QComboBox()
        self.q_combo_boundary_selector.addItems(["Dirichlet", "Neumann"])
        self.q_combo_boundary_selector.setFixedWidth(200)
        self.q_combo_boundary_selector.setEnabled(False)

        # directory selection for Pickle stream
        self.directory_selector_button = QPushButton("Select directory")

        # layout for general selection (scenario and options)
        layout_scenario = QGridLayout()
        layout_scenario.addWidget(QLabel("Scenario:"), 0, 0)
        layout_scenario.addWidget(self.q_combo_scenario_selector, 1, 0)
        layout_scenario.addWidget(QLabel("Boundary Conditions:"), 2, 0)
        layout_scenario.addWidget(self.q_combo_boundary_selector, 3, 0)
        layout_scenario.addWidget(self.directory_selector_button, 4, 0)

        layout_scenario.addWidget(dummy_placeholder_h, 1, 1)
        layout_scenario.addWidget(QLabel("Options:"), 0, 2)
        layout_scenario.addWidget(self.checkbox_show_grid, 1, 2)
        layout_scenario.addWidget(self.checkbox_plot_potential, 2, 2)
        layout_scenario.addWidget(self.checkbox_in_plane, 3, 2)
        layout_scenario.addWidget(self.checkbox_show_magnet, 4, 2)

        # selection of source distribution
        self.q_combo_source_selector = QComboBox()
        self.q_combo_source_selector.addItems(["Gaussian Unimodal (charge)",
                                               "Gaussian Bimodal (charge)",
                                               "Line Conductor (current density)",
                                               "Double Line Conductor (current density)",
                                               "Linear Current in-plane (current density)",
                                               "Circular Current in-plane (current density)",
                                               "Rectangular Permanent Magnet (remanent flux density)",
                                               "Horseshoe Magnet (remanent flux density)"])
        # selection of material distribution
        self.q_combo_material_selector = QComboBox()
        self.q_combo_material_selector.addItems(["Quadrants", "Strips"])

        # layout for source and material selectin
        layout_source_and_material = QVBoxLayout()
        layout_source_and_material.addWidget(dummy_placeholder_v)
        layout_source_and_material.addWidget(QLabel("Source:"))
        layout_source_and_material.addWidget(self.q_combo_source_selector)
        layout_source_and_material.addWidget(QLabel("Material Distribution:"))
        layout_source_and_material.addWidget(self.q_combo_material_selector)

        # material values in the four areas
        self.eps_1_line_edit = QLineEdit("1")  # permittivity
        self.eps_2_line_edit = QLineEdit("2")
        self.eps_3_line_edit = QLineEdit("3")
        self.eps_4_line_edit = QLineEdit("4")
        layout_eps = QGridLayout()
        layout_eps.addWidget(self.eps_1_line_edit, 0, 0)
        layout_eps.addWidget(self.eps_2_line_edit, 0, 1)
        layout_eps.addWidget(self.eps_3_line_edit, 1, 0)
        layout_eps.addWidget(self.eps_4_line_edit, 1, 1)

        self.mu_1_line_edit = QLineEdit("1")  # permeability
        self.mu_2_line_edit = QLineEdit("2")
        self.mu_3_line_edit = QLineEdit("3")
        self.mu_4_line_edit = QLineEdit("4")
        layout_mu = QGridLayout()
        layout_mu.addWidget(self.mu_1_line_edit, 0, 0)
        layout_mu.addWidget(self.mu_2_line_edit, 0, 1)
        layout_mu.addWidget(self.mu_3_line_edit, 1, 0)
        layout_mu.addWidget(self.mu_4_line_edit, 1, 1)
        index_mu = layout_mu.count()
        for i in range(index_mu):
            layout_mu.itemAt(i).widget().setEnabled(False)  # defaults to inactive before scenario is chosen

        # source values
        self.rho_line_edit = QLineEdit()
        self.j_line_edit = QLineEdit()
        self.B_line_edit = QLineEdit()

        # layout for value input
        layout_values = QGridLayout()
        self.eps_label = QLabel("Permittivity:")
        self.mu_label = QLabel("Permeability:")
        self.rho_label = QLabel("Charge Density:")
        self.j_label = QLabel("Current Density:")
        self.B_label = QLabel("Remanent Flux Density:")
        layout_values.addWidget(self.eps_label, 0, 0)
        layout_values.addLayout(layout_eps, 1, 0)
        layout_values.addWidget(self.mu_label, 0, 1)
        layout_values.addLayout(layout_mu, 1, 1)

        layout_values.addWidget(self.rho_label, 0, 2)
        layout_values.addWidget(self.rho_line_edit, 0, 3)
        layout_values.addWidget(self.B_label, 1, 2)
        layout_values.addWidget(self.B_line_edit, 1, 3)
        layout_values.addWidget(self.j_label, 2, 2)
        layout_values.addWidget(self.j_line_edit, 2, 3)

        text_edits = [self.eps_1_line_edit, self.eps_2_line_edit, self.eps_3_line_edit, self.eps_4_line_edit,
                      self.mu_1_line_edit, self.mu_2_line_edit, self.mu_3_line_edit, self.mu_4_line_edit,
                      self.rho_line_edit, self.j_line_edit, self.B_line_edit]
        value_labels = [self.eps_label, self.mu_label, self.rho_label, self.j_label, self.B_label]
        for te in text_edits:
            te.setFixedWidth(50)
            te.setValidator(QtGui.QDoubleValidator())  # forces the user to plug in a float value
            te.returnPressed.connect(self.on_click_line_edit)  # starts the FEM simulation when enter is pressed
        for sl in value_labels:
            sl.setFixedWidth(170)
        index_source = layout_values.count()
        for i in range(index_source):
            temp = layout_values.itemAt(i).widget()
            if temp is not None:
                temp.setEnabled(False)
        self.eps_label.setEnabled(True)
        self.rho_label.setEnabled(True)
        self.rho_line_edit.setEnabled(True)

        self.start_button = QPushButton("Start FEM")
        self.start_button.setFixedWidth(100)

        # layout for all user input
        layout_user_input = QGridLayout()
        layout_user_input.addLayout(layout_scenario, 0, 0)
        layout_user_input.addLayout(layout_source_and_material, 0, 1)
        layout_user_input.addWidget(dummy_placeholder_v)
        layout_user_input.addLayout(layout_values, 2, 0)
        layout_start_button = QGridLayout()
        layout_start_button.addWidget(dummy_placeholder_h, 0, 0)
        layout_start_button.addWidget(dummy_placeholder_v, 0, 1)
        layout_start_button.addWidget(dummy_placeholder_v, 1, 1)
        layout_start_button.addWidget(dummy_placeholder_v, 2, 1)
        layout_start_button.addWidget(self.start_button, 2, 1)
        layout_user_input.addLayout(layout_start_button, 2, 1)

        # top level layout
        layout_top_level = QVBoxLayout()
        layout_top_level.addLayout(layout_user_input)

        # FUNCTIONALITY
        # select scenario
        self.q_combo_scenario_selector.activated.connect(self.scenario_changed)
        self.selected_scenario = "Electrostatic"

        # select source
        self.q_combo_source_selector.activated.connect(self.source_changed)
        self.selected_source = "Gaussian Unimodal (charge)"

        # select material distribution
        self.q_combo_material_selector.activated.connect(self.material_changed)
        self.selected_material = "Quadrants"

        # select boundary type
        self.q_combo_boundary_selector.activated.connect(self.boundary_changed)
        self.boundary_type = "Dirichlet"

        # start calculation
        self.start_button.clicked.connect(self.start_fem)

        # read values
        self.eps_1_value = 1.0
        self.eps_2_value = 10.0
        self.eps_3_value = 1.0
        self.eps_4_value = 10.0
        self.eps_1_line_edit.textChanged.connect(self.eps_1_changed)
        self.eps_2_line_edit.textChanged.connect(self.eps_2_changed)
        self.eps_3_line_edit.textChanged.connect(self.eps_3_changed)
        self.eps_4_line_edit.textChanged.connect(self.eps_4_changed)

        self.mu_1_value = 1.0
        self.mu_2_value = 10.0
        self.mu_3_value = 1.0
        self.mu_4_value = 10.0
        self.mu_1_line_edit.textChanged.connect(self.mu_1_changed)
        self.mu_2_line_edit.textChanged.connect(self.mu_2_changed)
        self.mu_3_line_edit.textChanged.connect(self.mu_3_changed)
        self.mu_4_line_edit.textChanged.connect(self.mu_4_changed)

        self.rho_value = 1.0
        self.rho_line_edit.textChanged.connect(self.rho_changed)
        self.j_value = 1.0
        self.j_line_edit.textChanged.connect(self.j_changed)
        self.B_value = 1.0
        self.B_line_edit.textChanged.connect(self.B_changed)

        # options
        self.potential_checked = False
        self.in_plane_checked = False
        self.show_grid_checked = False
        self.show_magnet_checked = False
        self.checkbox_plot_potential.stateChanged.connect(self.checkbox_plot_potential_checked)
        self.checkbox_in_plane.stateChanged.connect(self.checkbox_in_plane_checked)
        self.checkbox_show_grid.stateChanged.connect(self.checkbox_show_grid_checked)
        self.checkbox_show_magnet.stateChanged.connect(self.checkbox_show_magnet_checked)
        self.simulate_new = True

        # select directory for pickle
        self.selected_directory = '/Users/florianmeiners/PycharmProjects'
        self.directory_selector_button.clicked.connect(self.call_directory_selector)

        # dummy plot
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.sc.axes.plot([0, 1, 2, 3, 4], np.random.rand(5,1))
        self.sc.axes.set_title("This could be your FEM simulation!")

        layout_top_level.addWidget(self.sc)
        self.line_output = QLineEdit("Ready.")
        self.line_output.setReadOnly(True)
        self.line_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_top_level.addWidget(self.line_output)

        # set window layout
        widget = QWidget()
        widget.setLayout(layout_top_level)

        self.setCentralWidget(widget)

    def scenario_changed(self):
        """
        Makes adjustments to the user interface according to the selected scenario. For example, if the magnetostatic
        case is selected, no input for the permittivity, current density, or charge density is accepted.

        Parameters:
        -----------
            None
        Returns:
        -----------
            nothing
        Sets:
        -----------
            self.checkbox_plot_potential.setEnabled, self.checkbox_in_plane.setEnabled, self.eps_label.setEnabled,
            self.mu_label.setEnabled, self.eps_i_line_edit.setEnabled, self.mu_i_line_edit.setEnabled (i = 1, ..., 4),
            self.rho_label.setEnabled, self.rho_line_edit.setEnabled, self.j_label.setEnabled, self.j_line_edit.setEnabled,
            self.B_label.setEnabled, self.B_line_edit.setEnabled, self.q_combo_boundary_selector.setEnabled, self.simulate_new
        """
        self.selected_scenario = self.q_combo_scenario_selector.currentText()
        if self.selected_scenario == "Electrostatic":
            self.simulate_new = True
            self.checkbox_plot_potential.setEnabled(True)
            self.checkbox_in_plane.setEnabled(False)
            self.checkbox_show_magnet.setEnabled(False)
            self.q_combo_boundary_selector.setEnabled(False)

            self.eps_label.setEnabled(True)
            self.eps_1_line_edit.setEnabled(True)
            self.eps_2_line_edit.setEnabled(True)
            self.eps_3_line_edit.setEnabled(True)
            self.eps_4_line_edit.setEnabled(True)

            self.mu_label.setEnabled(False)
            self.mu_1_line_edit.setEnabled(False)
            self.mu_2_line_edit.setEnabled(False)
            self.mu_3_line_edit.setEnabled(False)
            self.mu_4_line_edit.setEnabled(False)

            self.rho_label.setEnabled(True)
            self.rho_line_edit.setEnabled(True)
            self.j_label.setEnabled(False)
            self.j_line_edit.setEnabled(False)
            self.B_label.setEnabled(False)
            self.B_line_edit.setEnabled(False)

        elif self.selected_scenario == "Magnetostatic":
            self.simulate_new = True
            self.checkbox_plot_potential.setEnabled(False)
            self.checkbox_in_plane.setEnabled(False)
            self.checkbox_show_magnet.setEnabled(True)
            self.q_combo_boundary_selector.setEnabled(False)

            self.eps_label.setEnabled(False)
            self.eps_1_line_edit.setEnabled(False)
            self.eps_2_line_edit.setEnabled(False)
            self.eps_3_line_edit.setEnabled(False)
            self.eps_4_line_edit.setEnabled(False)

            self.mu_label.setEnabled(True)
            self.mu_1_line_edit.setEnabled(True)
            self.mu_2_line_edit.setEnabled(True)
            self.mu_3_line_edit.setEnabled(True)
            self.mu_4_line_edit.setEnabled(True)

            self.rho_label.setEnabled(False)
            self.rho_line_edit.setEnabled(False)
            self.j_label.setEnabled(False)
            self.j_line_edit.setEnabled(False)
            self.B_label.setEnabled(True)
            self.B_line_edit.setEnabled(True)

        elif self.selected_scenario == "Magnetoquasistatic":
            self.simulate_new = True
            self.checkbox_plot_potential.setEnabled(False)
            self.checkbox_in_plane.setEnabled(True)
            self.checkbox_show_magnet.setEnabled(False)

            self.eps_label.setEnabled(False)
            self.eps_1_line_edit.setEnabled(False)
            self.eps_2_line_edit.setEnabled(False)
            self.eps_3_line_edit.setEnabled(False)
            self.eps_4_line_edit.setEnabled(False)

            self.mu_label.setEnabled(True)
            self.mu_1_line_edit.setEnabled(True)
            self.mu_2_line_edit.setEnabled(True)
            self.mu_3_line_edit.setEnabled(True)
            self.mu_4_line_edit.setEnabled(True)

            self.rho_label.setEnabled(False)
            self.rho_line_edit.setEnabled(False)
            self.j_label.setEnabled(True)
            self.j_line_edit.setEnabled(True)
            self.B_label.setEnabled(False)
            self.B_line_edit.setEnabled(False)

    def source_changed(self):
        """
        Sets the selected source term for the FEM calculation.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.selected_source, self.simulate_new
        """
        self.simulate_new = True
        self.selected_source = self.q_combo_source_selector.currentText()

    def material_changed(self):
        """
        Sets the selected material distribution for the FEM calculation.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.selected_material, self.simulate_new
        """
        self.simulate_new = True
        self.selected_material = self.q_combo_material_selector.currentText()

    def boundary_changed(self):
        """
        Sets the desired boundary type for the FEM calculation.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.boundary_type, self.simulate_new
        """
        self.simulate_new = True
        self.boundary_type = self.q_combo_boundary_selector.currentText()

    def start_fem(self):
        """
        Starts the FEM calculation by setting up the material distribution and source term. If an inadequate source term
        is selected, an instruction is printed to the output line. The results are shown on the canvas by calling the
        respective plot functions.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.simulate_new
        """
        mesh_obj = Mesh2DRect(0.0, 2.0, 0.0, 1.0, nx=181, ny=91)
        # TODO: make it possible to pass parameter values from the user interface here!
        # TODO: change the demo functions and default values so everything looks pretty and sensible!

        # source term
        source = demo_functions.rho_gauss()
        match self.selected_source:
            case "Gaussian Unimodal (charge)":
                source = demo_functions.rho_gauss(Q=self.rho_value)
            case "Gaussian Bimodal (charge)":
                source = demo_functions.rho_gauss_2(Q=self.rho_value)
            case "Line Conductor (current density)":
                source = demo_functions.line_conductor()
            case "Double Line Conductor (current density)":
                source = demo_functions.line_conductor_2()
            case "Linear Current in-plane (current density)":
                source = demo_functions.make_line_current(p0=(0.5, 0.5), p1=(1.5, 0.5),
                                                          J0=self.j_value, thickness=0.01)
            case "Circular Current in-plane (current density)":
                source = demo_functions.make_circular_current(center=(1.0, 0.5), radius=0.2,
                                                              J0=-self.j_value, thickness=0.01)
            case "Rectangular Permanent Magnet (remanent flux density)":
                source = demo_functions.make_rectangular_Br(center=(1.0, 0.5), half_sizes=(0.2, 0.1), Br0=self.B_value,
                                                            direction=(1.0, 0.0), smoothing=0.01)
            case "Horseshoe Magnet (remanent flux density)":
                source = demo_functions.make_horseshoe_Br(center=(1.0, 0.5), leg_length=0.35, leg_thickness=0.08,
                                                          gap=0.10, yoke_thickness=0.08, Br0=self.B_value, angle_rad=0.0,
                                                          smoothing=0.01, opening="up", gap_flux_lr=+1)

        # material distribution
        material = demo_functions.mat_inhomogeneous()
        match self.selected_material:
            case "Quadrants":
                if self.selected_scenario == "Electrostatic":
                    material = demo_functions.mat_inhomogeneous(eps0=self.eps_1_value, eps1=self.eps_2_value,
                                                                eps2=self.eps_3_value, eps3=self.eps_4_value)
                else:
                    material = demo_functions.mat_inhomogeneous(eps0=self.mu_1_value, eps1=self.mu_2_value,
                                                                eps2=self.mu_3_value, eps3=self.mu_4_value)
            case "Strips":
                if self.selected_scenario == "Electrostatic":
                    material = demo_functions.mat_inhomogeneous_2(eps0=self.eps_1_value, eps1=self.eps_2_value,
                                                                  eps2=self.eps_3_value, eps3=self.eps_4_value)
                else:
                    material = demo_functions.mat_inhomogeneous_2(eps0=self.mu_1_value, eps1=self.mu_2_value,
                                                                  eps2=self.mu_3_value, eps3=self.mu_4_value)
        # scenario selection
        match self.selected_scenario:
            case "Electrostatic":
                if self.simulate_new:
                    try:
                        _, phi, Ex, Ey, _, nodes, tris = demo_maker.make_two_material_demo(mesh=mesh_obj,
                                                                                           charge_function=source,
                                                                                           mat_distribution=material)
                    except:
                        self.line_output.setText("Choose appropriate source term!")
                    else:
                        stream_data = {'Potential': phi, 'Nodes': nodes, 'Triangles': tris,
                                       'Electric Field x direction': Ex, 'Electric Field y direction': Ey}
                        try:
                            # use whatever directory you want
                            with open(self.selected_directory+'/simulation_stream.pkl', 'wb') as handle:
                                pickle.dump(stream_data, handle)
                        except:
                            self.line_output.setText("Choose existing directory!")

                        self.plot_electric_potential_and_field(nodes=nodes, tris=tris, phi=phi, Ex=Ex, Ey=Ey,
                                                               show_mesh=self.show_grid_checked,
                                                               plot_potential=self.potential_checked)
                        self.simulate_new = False
                else:
                    try:
                        # use whatever directory you want
                        with open(self.selected_directory+'/simulation_stream.pkl', 'rb') as handle:
                            streamed_data = pickle.load(handle)
                    except:
                        self.line_output.setText("Choose existing file or run with different parameters!")
                    else:
                        nodes, tris, phi = streamed_data['Nodes'], streamed_data['Triangles'], streamed_data['Potential']
                        Ex, Ey = streamed_data['Electric Field x direction'], streamed_data['Electric Field y direction']
                        self.plot_electric_potential_and_field(nodes=nodes, tris=tris, phi=phi, Ex=Ex, Ey=Ey,
                                                               show_mesh=self.show_grid_checked,
                                                               plot_potential=self.potential_checked)
            case "Magnetostatic":
                if self.simulate_new:
                    try:
                        _, A_z, Bx, By, _, nodes, tris = demo_maker.make_two_material_demo_magnetostatic(mesh_obj,
                                                                                            mat_distribution=material,
                                                                                            M_callable=source)
                    except:
                        self.line_output.setText("Choose appropriate source term!")
                    else:
                        stream_data = {'Mag. Pot.': A_z, 'Nodes': nodes, 'Triangles': tris,
                                       'Mag. Flux x direction': Bx, 'Mag. Flux y direction': By}
                        try:
                            # use whatever directory you want
                            with open(self.selected_directory+'/simulation_stream.pkl', 'wb') as handle:
                                pickle.dump(stream_data, handle)
                        except:
                            self.line_output.setText("Choose existing directory!")
                        self.plot_magnetic_field(nodes, tris, A_z, Bx, By, self.show_grid_checked)
                        if self.show_magnet_checked:
                            # TODO: this is dangerous, all parameters are passed whether the magnet is a horseshoe or not
                            # TODO: also, the parameters are independent of the source term right now, which makes no sense
                            self.plot_magnet_shape(center=(1.0, 0.5), half_sizes=(0.2, 0.1), angle_rad=0.0,
                                                   leg_length=0.35,
                                                   leg_thickness=0.08, gap=0.10, yoke_thickness=0.08, opening="up",
                                                   grid_n=600,
                                                   margin=0.25)
                        self.simulate_new = False
                else:
                    try:
                        # use whatever directory you want
                        with open(self.selected_directory+'/simulation_stream.pkl', 'rb') as handle:
                            streamed_data = pickle.load(handle)
                    except:
                        self.line_output.setText("Choose existing file or run with different parameters!")
                    else:
                        nodes, tris, phi = streamed_data['Nodes'], streamed_data['Triangles'], streamed_data['Mag. Pot.']
                        Bx, By = streamed_data['Mag. Flux x direction'], streamed_data['Mag. Flux y direction']
                        self.plot_magnetic_field(nodes, tris, phi, Bx, By, self.show_grid_checked)
                        if self.show_magnet_checked:
                            self.plot_magnet_shape(center=(1.0, 0.5), half_sizes=(0.2, 0.1), angle_rad=0.0,
                                                   leg_length=0.35,
                                                   leg_thickness=0.08, gap=0.10, yoke_thickness=0.08, opening="up",
                                                   grid_n=600,
                                                   margin=0.25)
            case "Magnetoquasistatic":
                if self.simulate_new:
                    if self.in_plane_checked:
                        try:
                            _, A_z, Bz, _, nodes, tris = demo_maker.make_two_material_demo_magnetic_nedelec(mesh_obj,
                                                                                    vector_source=source,
                                                                                    mat_distribution=material,
                                                                                    boundary_type=self.boundary_type)
                        except:
                            self.line_output.setText("Choose appropriate source term!")
                        else:
                            stream_data = {'Mag. Pot.': Bz, 'Nodes': nodes, 'Triangles': tris}
                            try:
                                # use whatever directory you want
                                with open(self.selected_directory+'/simulation_stream.pkl',
                                          'wb') as handle:
                                    pickle.dump(stream_data, handle)
                            except:
                                self.line_output.setText("Choose existing directory!")

                            self.plot_magnetic_flux_density_heatmap(nodes, tris, Bz, self.show_grid_checked)
                            if self.boundary_type == "Neumann":
                                self.line_output.setText(
                                    "Right now, Neumann boundary conditions correspond to a constant "
                                    "vector field (0,5).")
                            self.simulate_new = False
                    else:
                        try:
                            _, A_z, Bx, By, _, nodes, tris = demo_maker.make_two_material_demo_magnetic(mesh_obj,
                                                                                            current_function=source,
                                                                                            mat_distribution=material)
                        except:
                            self.line_output.setText("Choose appropriate source term!")
                        else:
                            stream_data = {'Mag. Pot.': A_z, 'Mag. Flux x direction': Bx, 'Mag. Flux y direction': By,
                                           'Nodes': nodes, 'Triangles': tris}
                            try:
                                with open(self.selected_directory+'/simulation_stream.pkl','wb') as handle:
                                    pickle.dump(stream_data, handle)
                            except:
                                self.line_output.setText("Choose existing directory!")
                            self.plot_magnetic_field(nodes, tris, A_z, Bx, By, self.show_grid_checked)
                            self.simulate_new = False
                else:
                    try:
                        with open(self.selected_directory+'/simulation_stream.pkl', 'rb') as handle:
                            streamed_data = pickle.load(handle)
                    except:
                        self.line_output.setText("Choose existing file or run with different parameters!")
                    else:
                        if self.in_plane_checked:
                            nodes, tris = streamed_data['Nodes'], streamed_data['Triangles']
                            phi = streamed_data['Mag. Pot.']
                            self.plot_magnetic_flux_density_heatmap(nodes, tris, phi, self.show_grid_checked)
                        else:
                            nodes, tris = streamed_data['Nodes'], streamed_data['Triangles']
                            phi = streamed_data['Mag. Pot.']
                            Bx, By = streamed_data['Mag. Flux x direction'], streamed_data['Mag. Flux y direction']
                            self.plot_magnetic_field(nodes, tris, phi, Bx, By, self.show_grid_checked)

    def eps_1_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.eps_1_value, self.simulate_new
        """
        if self.eps_1_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.eps_1_line_edit.setText(self.eps_1_line_edit.text())
            self.eps_1_value = float(self.eps_1_line_edit.text())

    def eps_2_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.eps_2_value, self.simulate_new
        """
        if self.eps_2_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.eps_2_line_edit.setText(self.eps_2_line_edit.text())
            self.eps_2_value = float(self.eps_2_line_edit.text())

    def eps_3_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.eps_3_value, self.simulate_new
        """
        if self.eps_3_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.eps_3_line_edit.setText(self.eps_3_line_edit.text())
            self.eps_3_value = float(self.eps_3_line_edit.text())

    def eps_4_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.eps_4_value, self.simulate_new
        """
        if self.eps_4_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.eps_4_line_edit.setText(self.eps_4_line_edit.text())
            self.eps_4_value = float(self.eps_4_line_edit.text())

    def mu_1_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.mu_1_value, self.simulate_new
        """
        if self.mu_1_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.mu_1_line_edit.setText(self.mu_1_line_edit.text())
            self.mu_1_value = float(self.mu_1_line_edit.text())

    def mu_2_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.mu_2_value, self.simulate_new
        """
        if self.mu_2_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.mu_2_line_edit.setText(self.mu_2_line_edit.text())
            self.mu_2_value = float(self.mu_2_line_edit.text())

    def mu_3_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.mu_3_value, self.simulate_new
        """
        if self.mu_3_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.mu_3_line_edit.setText(self.mu_3_line_edit.text())
            self.mu_3_value = float(self.mu_3_line_edit.text())

    def mu_4_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.mu_4_value, self.simulate_new
        """
        if self.mu_4_line_edit.hasAcceptableInput():
            self.mu_4_line_edit.setText(self.mu_4_line_edit.text())
            self.mu_4_value = float(self.mu_4_line_edit.text())

    def rho_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.rho_value, self.simulate_new
        """
        if self.rho_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.rho_line_edit.setText(self.rho_line_edit.text())
            self.rho_value = float(self.rho_line_edit.text())

    def j_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.j_value, self.simulate_new
        """
        if self.j_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.j_line_edit.setText(self.j_line_edit.text())
            self.j_value = float(self.j_line_edit.text())

    def B_changed(self):
        """
        Reads the user input for the respective parameter value.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.B_value, self.simulate_new
        """
        if self.B_line_edit.hasAcceptableInput():
            self.simulate_new = True
            self.B_line_edit.setText(self.B_line_edit.text())
            self.B_value = float(self.B_line_edit.text())

    def checkbox_show_grid_checked(self, s):
        """
        Selects whether to show the triangulation in the plot.

        Parameters:
        -----------
            s: int
                value of the checkbox_show_grid checkbox
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.show_grid_checked
        """
        if s == 0:
            self.show_grid_checked = False
        else:
            self.show_grid_checked = True

    def checkbox_plot_potential_checked(self, s):
        """
        Selects whether to plot the potential.

        Parameters:
        -----------
            s: int
                value of the checkbox_plot_potential checkbox
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.potential_checked
        """
        if s == 0:
            self.potential_checked = False
            if self.potential_window is not None:
                self.potential_window.close()
        else:
            self.potential_checked = True

    def checkbox_in_plane_checked(self, s):
        """
        Selects whether the current in the magnetoquasistatic case is in-plane.
        This also opens up the opportunity of applying Neumann boundary conditions.

        Parameters:
        -----------
            s: int
                value of the checkbox_in_plane checkbox
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.in_plane_checked, self.q_combo_boundary_selector.setEnabled, self.simulate_new
        """
        if s == 0:
            self.simulate_new = True
            self.in_plane_checked = False
            self.q_combo_boundary_selector.setEnabled(False)
        else:
            self.simulate_new = True
            self.in_plane_checked = True
            self.q_combo_boundary_selector.setEnabled(True)

    def checkbox_show_magnet_checked(self, s):
        """
        Selects whether the magnet in the magnetostatic case should be displayed in the plot.

        Parameters:
        -----------
            s: int
                value of the checkbox_show_magnet checkbox
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.magnet_checked
        """
        if s == 0:
            self.show_magnet_checked = False
        else:
            self.show_magnet_checked = True

    def on_click_line_edit(self):
        """
        Starts the FEM simulation at the press of the enter key when in the respective LineEdit field by calling
        self.start_fem().

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        """
        self.start_fem()

    def call_directory_selector(self):
        """
        Opens a file dialog for streaming directory selection at the press of the respective PushButton. Sets the
        directory for saving the pickled file.

        Parameters:
        -----------
            None
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.directory_selected
        """
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Directory")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if file_dialog.exec():
            self.selected_directory = file_dialog.selectedFiles()[0]

    def plot_electric_potential_and_field(self, nodes, tris, phi, Ex, Ey, show_mesh, plot_potential):
        """
        Plots the electric field and potential (if needed).

        Parameters:
        -----------
            nodes: np.ndarray
                nodes for the FEM calculation
            tris: np.ndarray
                triangles for the FEM calculation
            phi: np.ndarray
                value of the potential at the FEM nodes
            Ex: np.ndarray
                x component of the electric field at the FEM nodes
            Ey: np.ndarray
                y component of the electric field at the FEM nodes
            show_mesh: Boolean
                whether to show the mesh plot
            plot_potential: Boolean
                whether to plot the potential plot
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.sc
        """
        triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
        self.sc.axes.cla()
        self.sc.axes.tricontour(triobj, phi, levels=20, linewidths=0.8)
        step = max(1, len(nodes) // 3000)
        idx = np.where((np.sqrt(Ex[::step] ** 2 + Ey[::step] ** 2)) != 0)
        self.sc.axes.quiver(nodes[::step, 0][idx], nodes[::step, 1][idx],
                            Ex[::step][idx] / (np.sqrt(Ex[::step] ** 2 + Ey[::step] ** 2))[idx],
                            Ey[::step][idx] / (np.sqrt(Ex[::step] ** 2 + Ey[::step] ** 2))[idx],
                            [1 / (np.sqrt(Ex[::step] ** 2 + Ey[::step] ** 2))[idx]],
                            angles='xy', scale_units='xy', scale=50, cmap='cubehelix')
        self.sc.axes.set_title(r"Electric field $\mathbf{E}$ over contours of the potential $\varphi$")
        self.sc.axes.set_xlabel(r"$x$")
        self.sc.axes.set_ylabel(r"$y$")

        # plot potential if desired
        if plot_potential:
            self.potential_window = PotentialWindow()
            self.potential_window.potential_plot_canvas.axes.tricontourf(triobj, phi, levels=30, cmap='cubehelix')
            self.potential_window.potential_plot_canvas.axes.set_title(r"Electrostatic Potential $\varphi$")
            self.potential_window.potential_plot_canvas.axes.set_xlabel("x")
            self.potential_window.potential_plot_canvas.axes.set_ylabel("y")
            self.potential_window.potential_plot_canvas.draw()
            self.potential_window.show()

        # plot mesh if desired
        if show_mesh:
            self.sc.axes.triplot(triobj, linewidth=0.4, alpha=0.4, color="black")
            if plot_potential:
                self.potential_window.potential_plot_canvas.axes.triplot(triobj, linewidth=0.4, alpha=0.4, color="black")
        self.sc.draw()

    def plot_magnetic_field(self, nodes, tris, A_v, Bx, By, show_mesh):
        """
        Plots the magnetic field.

        Parameters:
        -----------
            nodes: np.ndarray
                nodes for the FEM calculation
            tris: np.ndarray
                triangles for the FEM calculation
            A_v: np.ndarray
                value of the magnetic vector potential at the FEM nodes
            Bx: np.ndarray
                x component of the magnetic field at the FEM nodes
            By: np.ndarray
                y component of the magnetic field at the FEM nodes
            show_mesh: Boolean
                whether to show the mesh plot
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.sc
        """
        if self.potential_window is not None:
            self.potential_window.close()
        triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
        self.sc.axes.cla()
        self.sc.axes.tricontour(triobj, A_v, levels=20, linewidths=0.8, cmap='cubehelix')
        step = max(1, len(nodes)//3000)
        idx = np.where((np.sqrt(Bx[::step] ** 2 + By[::step] ** 2)) != 0)
        self.sc.axes.quiver(nodes[::step, 0][idx], nodes[::step, 1][idx],
                            Bx[::step][idx] / (np.sqrt(Bx[::step] ** 2 + By[::step] ** 2))[idx],
                            By[::step][idx] / (np.sqrt(Bx[::step] ** 2 + By[::step] ** 2))[idx],
                            [1 / (np.sqrt(Bx[::step] ** 2 + By[::step] ** 2))[idx]],
                            angles='xy', scale_units='xy', scale=50, cmap='cubehelix')
        self.sc.axes.set_title(r"Magnetic field $\mathbf{B}$ ")
        self.sc.axes.set_xlabel(r"$x$")
        self.sc.axes.set_ylabel(r"$y$")

        if show_mesh:
            self.sc.axes.triplot(triobj, linewidth=0.4, alpha=0.4, color="black")
        self.sc.draw()

    def plot_magnetic_flux_density_heatmap(self, nodes, tris, Bz_cells, show_mesh):
        """
        Plots the out-of-plane component of the magnetic field.

        Parameters:
        -----------
            nodes: np.ndarray
                nodes for the FEM calculation
            tris: np.ndarray
                triangles for the FEM calculation
            Bz_cells: np.ndarray
                z component of the magnetic field at the FEM nodes
            show_mesh: Boolean
                whether to show the mesh plot
        Returns:
        -----------
            Nothing
        Sets:
        -----------
            self.sc
        """
        if self.potential_window is not None:
            self.potential_window.close()
        triobj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
        self.sc.axes.cla()
        self.sc.axes.tripcolor(triobj, Bz_cells, shading="flat", cmap='cubehelix')
        self.sc.axes.set_title(r"Out-of-plane magnetic flux density $\mathbf{B}_z$")
        self.sc.axes.set_xlabel(r"$x$")
        self.sc.axes.set_ylabel(r"$y$")

        if show_mesh:
            self.sc.axes.triplot(triobj, linewidth=0.4, alpha=0.4, color="black")
        self.sc.draw()

    def plot_magnet_shape(self, center, half_sizes, angle_rad, leg_thickness, leg_length, gap, yoke_thickness, margin,
                          grid_n, opening):
        """
        Shows the shape of the permanent magnet in the magnetostatic case.

        Parameters:
        -----------
            center: np.ndarray
                center of the magnet in x,y coordinates
            half_sizes: np.ndarray
                half sizes of the magnet
            angle_rad: float
                angle of the magnet relative to the horizontal line
            leg_thickness: float
                leg thickness of a horseshoe magnet
            leg_length: float
                leg length of a horseshoe magnet
            gap: float
                gap width of a horseshoe magnet
            yoke_thickness: float
                yoke thickness of a horseshoe magnet
            margin: float
                allowed margin for an area with nonzero remanent flux density to be considered part of the magnet
            grid_n: int
                number of grid points to plot the outline from
            opening: string
                direction in which a horseshow magnet is pointing

        Returns:
        -----------
            Nothing
        """
        match self.selected_source:
            case "Rectangular Permanent Magnet (remanent flux density)":
                x1_unrotated_untranslated = np.array([- half_sizes[0], - half_sizes[1]])
                x2_unrotated_untranslated = np.array([- half_sizes[0], + half_sizes[1]])
                x3_unrotated_untranslated = np.array([+ half_sizes[0], + half_sizes[1]])
                x4_unrotated_untranslated = np.array([+ half_sizes[0], - half_sizes[1]])

                x_unrotated_untranslated = [x1_unrotated_untranslated, x2_unrotated_untranslated,
                                            x3_unrotated_untranslated, x4_unrotated_untranslated]
                x_untranslated = np.zeros_like(x_unrotated_untranslated)
                rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
                for i in range(len(x_unrotated_untranslated)):
                    x_untranslated[i] = rot_mat.dot(x_unrotated_untranslated[i])

                x_fin = np.zeros_like(x_untranslated)
                for i in range(len(x_untranslated)):
                    x_fin[i] = x_untranslated[i] + np.array([center[0], center[1]])

                self.sc.axes.plot(np.pad(x_fin[:, 0], (0, 1), mode="wrap"),
                                  np.pad(x_fin[:, 1], (0, 1), mode="wrap"), color="red", linewidth=1)
                self.sc.draw()
            case "Horseshoe Magnet (remanent flux density)":
                t = float(leg_thickness)
                L = float(leg_length)
                g = float(gap)
                ty = float(yoke_thickness)
                outer_w = g + 2.0 * t
                total_h = ty + L
                half_w = 0.5 * outer_w
                half_h = 0.5 * total_h
                R = np.hypot(half_w, half_h) + margin

                xc, yc = center
                x = np.linspace(xc - R, xc + R, int(grid_n))
                y = np.linspace(yc - R, yc + R, int(grid_n))
                X, Y = np.meshgrid(x, y)
                # calls the signed distance to the horseshoe in demo_functions
                SD = demo_functions.horseshoe_signed_distance(X, Y, center=center, leg_length=leg_length,
                                                              leg_thickness=leg_thickness, gap=gap, opening=opening,
                                                              yoke_thickness=yoke_thickness, angle_rad=angle_rad)
                self.sc.axes.contour(X, Y, SD, levels=[0.0], colors="red", linewidths=1)
                self.sc.draw()

    def closeEvent(self, event):  # this closes all windows
        """
        Closes all windows if the main window is closed.
        """
        QApplication.closeAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()