"""
=================================================================
FINITE ELEMENT METHOD (FEM) SOLVER FOR ALLEN-CAHN EQUATION (2D)
=================================================================
This script simulates the motion of phase boundaries (interfaces)
using the Cahn-Hilliard equation on a unit square domain.

PDE:
1: Cahn-Hilliard equation in mixed formulation form
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
   

def cahn_hilliard(
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
    T = final_time                                                      #Total simulation time (T)
    dt = T/nb_of_time_steps                                    
    M = 0.1                                                                                
    t = 0.0 


    """
    =============================================================
    2. DOMAIN, MESH GENERATION AND FINITE ELEMENT SPACE
    =============================================================
    """

    msh = create_unit_square(MPI.COMM_WORLD, nb_of_spatial_steps, nb_of_spatial_steps, CellType.triangle)
    P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
    #V = functionspace(msh, P1)
    ME = functionspace(msh, mixed_element([P1, P1]))



    """
    =================================================================
    3. INITIAL CONDITIONS
    =================================================================
    """


    solution = Function(ME)  # current solution
    solution_previous_time = Function(ME)
    solution_previous_iteration = Function(ME)
    initial = Function(ME)  # solution from previous converged step

    rng = np.random.default_rng(42)
    #initial.sub(0).interpolate(lambda x: 0.02 * (0.5 - rng.random(x.shape[1])))
    initial.sub(0).interpolate(lambda x: 0.02 * ( rng.random(x.shape[1])))
    initial.x.scatter_forward()


    """
    =================================================================
    4. Weak formulation of each linear elliptic problem 
    =================================================================
    """
    # Split mixed functions


    u, mu = ufl.split(solution)
    u_previous_time, mu_previous_time = ufl.split(solution_previous_time)
    u_previous_iteration, mu_previous_iteration = ufl.split(solution_previous_iteration)
    u0, mu0 = ufl.split(initial)

    u_trial, mu_trial = ufl.TrialFunctions(ME)
    phi, v = ufl.TestFunctions(ME)

    # 2. Set up your linearization constant (L)
    if type_of_linearisation == "newton":
        L = 3 * solution_previous_iteration**2
    elif type_of_linearisation == "L":
        L = dolfinx.fem.Constant(msh, dolfinx.default_scalar_type(3.0))


    # 3. Define the equations using the TrialFunctions
    F1 = (
        ufl.inner(u_trial, phi)*ufl.dx 
        + dt * ufl.inner(ufl.grad(mu_trial), ufl.grad(phi))*ufl.dx 
        - ufl.inner(u_previous_time, phi)*ufl.dx
    )

    F2 = (
        -eps * ufl.inner(ufl.grad(u_trial), ufl.grad(v))*ufl.dx 
        + ufl.inner(mu_trial, v)*ufl.dx
        - L * ufl.inner(u_trial, v)*ufl.dx 
        - ufl.inner(u_previous_iteration**3, v)*ufl.dx 
        + L * ufl.inner(u_previous_iteration, v)*ufl.dx 
        + ufl.inner(u_previous_time, v)*ufl.dx
    )

    F = F1+F2

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
        u=solution, 
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

    V0, _ = ME.sub(0).collapse()
    topology, cell_types, x = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    c_visual = Function(V0)
    c_visual.x.array[:] = solution.sub(0).collapse().x.array

    
    grid.point_data["c"] = c_visual.x.array.real
    grid.set_active_scalars("c")  # Aangepast naar string "c" ipv UFL object c

  
    # --- 3. Initialize the VTK Writer ---
    solution.name = "c"
    vtk_filepath = os.path.join("graphs", "output_solution.pvd")
    vtk_file = dolfinx.io.VTKFile(msh.comm, vtk_filepath, "w")
    c_output = dolfinx.fem.Function(V0, name="c")


    """
    =================================================================
    6. Time stepping and iteration loop
    =================================================================
    """

    errors = []
    energies = []
    iterations = []

    solution_previous_time.x.array[:] = initial.x.array[:]
    solution_previous_iteration.x.array[:] = initial.x.array[:]
    step = 0
    error = 1
    nb_of_iterations = 0
    while t < T:
        t += dt
        while error > 10**(-8):
            _ = problem.solve()
            print(solution.x.array[:])

            compiled_error = form(ufl.inner(u - u_previous_iteration, u - u_previous_iteration) * ufl.dx)
            error = np.sqrt(assemble_scalar(compiled_error))
            errors.append(error)
            print(error)

            compiled_energy = form((0.5*ufl.inner(ufl.grad(u), ufl.grad(u))+(0.25/eps)*(1-u**2)**2)*ufl.dx)
            energy = assemble_scalar(compiled_energy)
            energies.append(energy)
            print(energy)

            solution_previous_iteration.x.array[:] = solution.x.array[:]
            nb_of_iterations += 1
        
        iterations.append(nb_of_iterations)
        nb_of_iterations = 0
        
        solution_previous_time.x.array[:] = solution.x.array[:]
        solution_previous_iteration.x.array[:] = solution.x.array[:]

        c_output.interpolate(solution.sub(0))
        vtk_file.write_function(c_output, t)

        error = 1
        step += 1
        print("step = " + str(step))


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

    # Write energies to energies.txt
    with open(iterations_path, "w") as f:
        for iteration in iterations:
            f.write(f"{iteration}\n")

    vtk_file.close()
    return iterations, errors, energies



    

