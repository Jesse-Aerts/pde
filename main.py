import cahn_hilliard as ch
import allen_cahn as ac
from plot import plot2d, plot3d, animation, plot_errors, plot_energies


"""
iterations, errors, energies = ac.allen_cahn(
    type_of_linearisation = "L",              
    nb_of_spatial_steps = 100,                                  
    nb_of_time_steps=10,  
    final_time = 10000, 
    eps= 0.05
)

"""


iterations, errors, energies = ch.cahn_hilliard(
    type_of_linearisation = "newton",              
    nb_of_spatial_steps = 100,                                  
    nb_of_time_steps=100,  
    final_time = 5, 
    eps = 0.05  
)


animation()


plot_errors(False, False,[1,2,3,4,5,6,7,8,9,10])

plot_energies(False, False, [1,2,3,4,5,6,7,8,10])

