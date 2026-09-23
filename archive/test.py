import dolfinx.fem
import ufl
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import CellType, create_unit_interval
from mpi4py import MPI
import numpy as np

# 1. Maak een heel simpel 1D domein met 2 elementen
msh = create_unit_interval(MPI.COMM_WORLD, 2)
V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

# 2. Maak een variabele 'L_val' die de matrix zou moeten veranderen
L_val = dolfinx.fem.Function(V)
L_val.x.array[:] = 1.0  # We beginnen met L_val = 1.0

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# De matrix-vorm A hangt direct af van L_val:
a = L_val * ufl.inner(u, v) * ufl.dx
f = dolfinx.fem.Constant(msh, 1.0) * v * ufl.dx

# 3. Maak het LinearProblem aan
sol = dolfinx.fem.Function(V)
problem = LinearProblem(a, f, u=sol, petsc_options_prefix = "H")

# --- TEST 1: Los op met L_val = 1.0 ---
_ = problem.solve()
print("Oplossing 1 (met L_val = 1.0):", sol.x.array[0])

# --- TEST 2: Verander L_val nu naar 100.0! ---
# Als de matrix opnieuw geassembleerd wordt, moet de oplossing 100x KLEINER worden.
L_val.x.array[:] = 100.0

_ = problem.solve()
print("Oplossing 2 (met L_val = 100.0):", sol.x.array[0])