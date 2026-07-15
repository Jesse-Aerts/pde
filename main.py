import cahn_hilliard as ch
import allen_cahn as ac
from plot import plot2d, plot3d, animation, plot_errors, plot_energies, plot_errors2, plot_energies2, plot_errors3, plot_errors4, plot_energies4


errors, energies = ac.allen_cahn(
    type_of_linearisation = "newton",              #Scheme: "newton" (quadratic convergence), "L" (simple linear), or "M" (modified stabilized)
    nb_of_spatial_steps = 50,                                   #Mesh resolution (number of segments per boundary edge)
    nb_of_time_steps=100,                                       #Total number of time steps (N_T)
    eps = 10**(-4)                                              #Interfacial width parameter (epsilon^2)
)
#errors, energies = ch.cahn_hilliard()

"""
plot2d(5)
plot3d(3)
animation()
"""

plot_errors4([1,2,3])

plot_energies4([2,3,4,8])

