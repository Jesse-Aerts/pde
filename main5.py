import cahn_hilliard as ch
import plot2 as pm  # Pas aan naar de naam van jouw plot-bestand

# 1. Definieer de waarden voor epsilon die je wilt vergelijken
eps_values = [0.7,0.6,0.05,0.04]
time_values =[1,0.1,0.01,0.001]

# Lijsten om de resultaten per eps-waarde op te slaan
all_iterations = []
all_errors = []
all_energies = []
all_modified_energies = []
labels = []

# 2. Lus over de eps-waarden en voer de simulaties uit
for time in time_values:
    print(f"Simulatie starten voor eps = {time}...")
    
    iterations, errors, energies, modified_energies = ch.cahn_hilliard(
        type_of_linearisation="newton",              
        nb_of_spatial_steps=100,                                   
        nb_of_time_steps=1,   
        final_time=time, 
        eps=0.05
    )
    
    # Resultaten verzamelen
    all_iterations.append(iterations)
    all_errors.append(errors)
    all_energies.append(energies)
    all_modified_energies.append(modified_energies)
    labels.append(rf"$\Delta t = {time}$")

# 3. Genereer de vergelijkende plots
print("\nPlots genereren...")

# Plot 1: H1-norm error convergentie per eps
pm.plot_errors(
    iterations_per_timestep=all_iterations,
    errors=all_errors,
    timesteps=1,
    labels=labels
)

# Plot 2: Ginzburg-Landau energie E(u) per eps
pm.plot_energies(
    iterations_per_timestep=all_iterations,
    energies=all_energies,
    timesteps=1,
)

# Plot 3: Aangepaste functionaal J(u) per eps
pm.plot_modified_energies(
    iterations_per_timestep=all_iterations,
    energies=all_modified_energies,
    timesteps=1,
)

