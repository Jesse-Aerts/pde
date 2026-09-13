import cahn_hilliard as ch
import plot2 as pm  # Pas aan naar de naam van jouw plot-bestand

# 1. Definieer de waarden voor epsilon die je wilt vergelijken
time_values =[10]

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
        type_of_linearisation="L",              
        nb_of_spatial_steps=50,                                   
        nb_of_time_steps=1,   
        final_time=time, 
        eps=0.06
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


# Plot 3: Ginzburg-Landau energie E(u) voor de eerste run 
pm.plot_energies(
    iterations_per_timestep=all_iterations[0],
    energies=all_energies[0],
    timesteps=[1]
)

# Plot 4: Aangepaste functionaal J(u) voor de eerste run (
pm.plot_modified_energies(
    iterations_per_timestep=all_iterations[0],
    energies=all_modified_energies[0],
    timesteps=[1]
)
