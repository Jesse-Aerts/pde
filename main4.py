import cahn_hilliard as ch
import plot2 as pm  # Pas aan naar de naam van jouw plot-bestand

# 1. Definieer de waarden voor epsilon die je wilt vergelijken
eps_values = [0.09, 0.08, 0.07,0.06]
eps_values = [0.07]

# Lijsten om de resultaten per eps-waarde op te slaan
all_iterations = []
all_errors = []
all_energies = []
all_modified_energies = []
labels = []

# 2. Lus over de eps-waarden en voer de simulaties uit
for eps in eps_values:
    print(f"Simulatie starten voor eps = {eps}...")
    
    iterations, errors, energies, modified_energies = ch.cahn_hilliard(
        type_of_linearisation="newton",              
        nb_of_spatial_steps=50,                                   
        nb_of_time_steps=100,   
        final_time=0.001, 
        eps=eps
    )
    
    # Resultaten verzamelen
    all_iterations.append(iterations)
    all_errors.append(errors)
    all_energies.append(energies)
    all_modified_energies.append(modified_energies)
    labels.append(rf"$\varepsilon = {eps}$")

# 3. Genereer de plots
print("\nPlots genereren...")

# Plot 1: H1-norm error convergentie per eps (semi-log plot: fout vs iteratie i)
pm.plot_errors(
    iterations_per_timestep=all_iterations,
    errors=all_errors,
    timesteps=1,
    labels=labels
)

# Plot 2: Kwadratische convergentie check (log-log plot: e_i vs e_{i-1})
pm.plot_quadratic_convergence(
    iterations_per_timestep=all_iterations,
    errors=all_errors,
    timesteps=1,
    labels=labels
)

# Plot 3: Ginzburg-Landau energie E(u) voor de eerste run (eps = 0.3)
pm.plot_energies(
    iterations_per_timestep=all_iterations[0],
    energies=all_energies[0],
    timesteps=1,
)

# Plot 4: Aangepaste functionaal J(u) voor de eerste run (eps = 0.3)
pm.plot_modified_energies(
    iterations_per_timestep=all_iterations[0],
    energies=all_modified_energies[0],
    timesteps=1,
)