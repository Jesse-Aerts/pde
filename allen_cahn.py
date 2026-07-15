"""
=================================================================
FINITE ELEMENT METHOD (FEM) SOLVER FOR ALLEN-CAHN EQUATION (2D)
=================================================================
This script simulates the motion of phase boundaries (interfaces)
using the Cahn-Hilliard equation on a unit square domain.

PDE:
1: Allen Cahn equation
2: Boundary Conditions: homogeneous Neumann
3: Initial condition: ...   

Numerical Method:
1. Convex-Concave split of the potential term
2. Time Discretization: First order IMEX Euler scheme.
3. Tackling nonlinearity with an iterative linearization: Newton or L-scheme
4. Spatial Discretization: FEM with P1 Finite Elements (CG)
=================================================================
"""


"""
=============================================================
0. Importing modules & packages
=============================================================
"""

import os
from pathlib import Path
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace, form, assemble_scalar
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.io
import pyvista as pv
import pyvistaqt as pvqt
import shutil
   

def allen_cahn(
    type_of_linearisation = "newton", 
    nb_of_spatial_steps =50, 
    nb_of_time_steps = 100, 
    final_time = 1, 
    eps = 10**(-4)):
    """
    =============================================================
    1. PARAMETER DEFINITIONS
    =============================================================
    """

    #type_of_linearisation = type_of_linearisation               #Scheme: "newton" (quadratic convergence), "L" (simple linear), or "M" (modified stabilized)
    #nb_of_spatial_steps = 50                                    #Mesh resolution (number of segments per boundary edge)
    #nb_of_time_steps=100                                        #Total number of time steps (N_T)
    T = final_time                                                      #Total simulation time (T)
    dt = T/nb_of_time_steps                                     #Time step size (dt)
    #eps = 10**(-4)                                              #Interfacial width parameter (epsilon^2)
    M = 0.1                                                     #Stabilization constant used only in the "M" scheme                                
    t = 0.0 


    """
    =============================================================
    2. DOMAIN, MESH GENERATION AND FINITE ELEMENT SPACE
    =============================================================
    """

    msh = create_unit_square(MPI.COMM_WORLD, nb_of_spatial_steps, nb_of_spatial_steps, CellType.triangle)
    P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
    V = functionspace(msh, P1)


    """
    =================================================================
    3. INITIAL CONDITIONS
    =================================================================
    """


    u0 = Function(V)  # solution from the previous converged step
    solution_previous_time = Function(V)                    #Solution from time t_n (u^n)
    solution_previous_iteration = Function(V)

    rng = np.random.default_rng(42)
    u0.interpolate(lambda x: 0.02 * (0.5 - rng.random(x.shape[1])))
    u0.x.scatter_forward()   #this is needed for parallelcomputing



    """
    =================================================================
    4. Weak formulation of each linear elliptic problem 
    =================================================================
    """
    u = Function(V)  

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    if type_of_linearisation == "newton":
        L = 3*solution_previous_iteration**2
    elif type_of_linearisation == "L":
        L = 3


    F = (
        dt * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx 
        + (1 + (dt * L) / eps) * ufl.inner(u_trial, v_test) * ufl.dx 
        - (dt * L) / eps * ufl.inner(solution_previous_iteration, v_test) * ufl.dx 
        + (dt / eps) * ufl.inner(solution_previous_iteration**3, v_test) * ufl.dx 
        - ((dt / eps) + 1) * ufl.inner(solution_previous_time, v_test) * ufl.dx
    )


    a = ufl.lhs(F)
    f_linear = ufl.rhs(F)


    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
        "snes_atol": 0,
        "snes_rtol": 0,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "petsc",
        "snes_monitor": None,
    }

    problem = LinearProblem(
        a, 
        f_linear, 
        u=u, 
        petsc_options=petsc_options, 
        petsc_options_prefix= "demo_helmholtz_"
    )


    """
    =================================================================
    5. Output preperation
    =================================================================
    """


    # --- 1. Clean and Prepare the Output Directory ---
    if msh.comm.rank == 0:
        # If the folder already exists, delete it and all its contents
        if os.path.exists("graphs"):
            shutil.rmtree("graphs")
        if os.path.exists("outputs"):
            shutil.rmtree("outputs")
        # Create a fresh, empty directory
        os.makedirs("graphs", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

    # MPI Barrier: Forces all processor ranks to wait until rank 0 
    # is done deleting and recreating the directory.
    msh.comm.Barrier()

    # --- 2. PyVista Mesh and Plotter Setup ---
    topology, cell_types, x = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    c_visual = Function(V)
    c_visual.x.array[:] = u.x.array

    grid.point_data["c"] = c_visual.x.array.real
    grid.set_active_scalars("c")

    # --- 3. Initialize the VTK Writer ---
    u.name = "c"
    vtk_filepath = os.path.join("graphs","output_solution.pvd")
    vtk_file = dolfinx.io.VTKFile(msh.comm, vtk_filepath, "w")


    """
    =================================================================
    6. Time stepping and iteration loop
    =================================================================
    """

    errors = []
    energies = []
    iterations = []

    solution_previous_time.x.array[:] = u0.x.array[:]
    solution_previous_iteration.x.array[:] = u0.x.array[:]
    step = 1
    error = 1
    nb_of_iterations = 0
    while t < T:
        t += dt
        while error > 10**(-8):
            _ = problem.solve()

            compiled_error = form(ufl.inner(u - solution_previous_iteration, u - solution_previous_iteration) * ufl.dx)
            error = np.sqrt(assemble_scalar(compiled_error))
            errors.append(error)
            print(error)

            compiled_energy = form((0.5*ufl.inner(ufl.grad(u), ufl.grad(u))+(0.25/eps)*(1-u**2)**2)*ufl.dx)
            energy = assemble_scalar(compiled_energy)
            energies.append(energy)
            print(energy)

            solution_previous_iteration.x.array[:] = u.x.array[:]
            nb_of_iterations += 1
        
        solution_previous_time.x.array[:] = u.x.array[:]
        solution_previous_iteration.x.array[:] = u.x.array[:]

        iterations.append(nb_of_iterations)
        nb_of_iterations = 0

        vtk_file.write_function(u, t)

        error = 1
        step += 1
        print("step = " + str(step))

    vtk_file.close()
    errors_path = os.path.join("outputs", "errors.txt")
    energies_path = os.path.join("outputs", "energies.txt")
    iterations_path = os.path.join("outputs", "iterations.txt")

    # Write errors to errors.txt
    with open(errors_path, "w") as f:
        for error in errors:
            f.write(f"{error}\n")

    # Write energies to energies.txt
    with open(energies_path, "w") as f:
        for energy in energies:
            f.write(f"{energy}\n")
    vtk_file.close()

    with open(iterations_path, "w") as f:
        for iteration in iterations:
            f.write(f"{iteration}\n")



    return iterations, errors, energies



    

