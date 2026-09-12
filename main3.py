import allen_cahn as ac
from plot import plot2d, plot3d, animation, plot_errors, plot_energies, plot_combined_energies, plot_modified_energies
import matplotlib.pyplot as plt

# --- 1. Newton meting ---
start_cpu_newton = time.process_time()

iterations_newton, errors_newton, energies_newton, modified_energies_newton = ch.cahn_hilliard(
    type_of_linearisation = "newton",              
    nb_of_spatial_steps = 200,                                   
    nb_of_time_steps = 2,   
    final_time = 1, 
    eps = 0.05
)

end_cpu_newton = time.process_time()
cpu_time_newton = end_cpu_newton - start_cpu_newton


# --- 2. L-scheme (LU) meting ---
start_cpu_L = time.process_time()

iterations_L, errors_L, energies_L, modified_energies_L = chl.cahn_hilliard(
    type_of_linearisation = "L",              
    nb_of_spatial_steps = 100,                                   
    nb_of_time_steps = 1,   
    final_time = 1, 
    eps = 0.1
)

end_cpu_L = time.process_time()
cpu_time_L = end_cpu_L - start_cpu_L


# --- 3. Resultaten printen ---
print("\n" + "="*50)
print("BENCHMARK RESULTATEN (CPU-tijd):")
print(f"Newton CPU-tijd:   {cpu_time_newton:.4f} seconden")
print(f"L-scheme CPU-tijd: {cpu_time_L:.4f} seconden")

if cpu_time_L > 0:
    speedup = cpu_time_newton / cpu_time_L
    print(f"Speedup L-scheme:  {speedup:.2f}x sneller dan Newton")
print("="*50 + "\n")
"""
plot_errors(False, False,[1])

plot_errors(False, False,[1,2,3,4,5,6,7])


plot_energies(False, False, [1])

plot_energies(False, False, [1,2,3,4,5,6,7])


plot_modified_energies(False, False, [1])


plot_modified_energies(False, False, [1,2,3,4,5,6,7])




iterations, errors, energies = ac.allen_cahn(
    type_of_linearisation = "L",              
    nb_of_spatial_steps = 50,                                  
    nb_of_time_steps=10,  
    final_time = 10, 
    eps= 0.05
)





plot_errors(False, False,[1])

plot_errors(False, False,[1,2,3,4,5,6,7])


plot_energies(False, False, [1])

plot_energies(False, False, [1,2,3,4,5,6,7])


plot_modified_energies(False, False, [1])


plot_modified_energies(False, False, [1,2,3,4,5,6,7])



error = []
for i in range(1, 10):
    iterations, errors, energies, abs_errors = cht.cahn_hilliard_test(
        type_of_linearisation = "newton",              
        nb_of_spatial_steps = 10*i,                                  
        nb_of_time_steps=20,  
        final_time = 1, 
        eps = 1
    )
    error.append(abs_errors[10])
    print(error)


plt.plot(error)
plt.xlabel("Step")
plt.ylabel("Value")
plt.grid(True)
plt.show()


animation()


plot_errors(False, False,[1,2,3,4,5,6,7,8,9,10])

plot_energies(False, False, [1,2,3,4,5,6,7,8,10])
"""

"""
import matplotlib.pyplot as plt
import numpy as np

# Jouw data
h_values = np.array([1/10, 1/20, 1/30, 1/40, 1/50, 1/60, 1/70, 1/80, 1/90])
errors   = np.array([
    0.046520295077165784, 
    0.012030363543963682, 
    0.005382763181341247, 
    0.003035160178687071, 
    0.0019447364758137127, 
    0.00135137010346318, 
    0.0009932295080151193, 
    0.0007606358941163737, 
    0.0006011020248808212
])

# Maak de log-log plot
plt.figure(figsize=(7, 5))
plt.loglog(h_values, errors, 'o-', label="Berekende L2-fout", color='navy', linewidth=2)

# Voeg een theoretische O(h^2) referentielijn toe
# C * h^2 waarbij we C afstemmen op het eerste punt
C = errors[0] / (h_values[0]**2)
plt.loglog(h_values, C * (h_values**2), '--', label=r"Theoretisch $\mathcal{O}(h^2)$", color='crimson')

# Opmaak
plt.xlabel(r"Maaswijdte $h$", fontsize=12)
plt.ylabel(r"$L_2$-fout", fontsize=12)
plt.title("Convergentie-analyse Cahn-Hilliard ($P_1$ elementen)", fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

# Toon of bewaar de grafiek
plt.savefig("cahn_hilliard_convergence.png", dpi=300)
plt.show()
"""