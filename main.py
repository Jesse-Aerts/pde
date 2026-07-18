import cahn_hilliard as ch
import allen_cahn as ac
from plot import plot2d, plot3d, animation, plot_errors, plot_energies



iterations, errors, energies = ac.allen_cahn(
    type_of_linearisation = "newton",              
    nb_of_spatial_steps = 50,                                  
    nb_of_time_steps=2,  
    final_time = 100000000, 
    eps = 10**(-2)
)

"""
iterations, errors, energies = ch.cahn_hilliard(
    type_of_linearisation = "newton",              
    nb_of_spatial_steps = 50,                                  
    nb_of_time_steps=100,  
    final_time = 1000, 
    eps = 10**(-3)   
)
"""

animation()


plot_errors(False, False,[0,1,2,3,4])

plot_energies(False, False, [0,1,2,3,4])

