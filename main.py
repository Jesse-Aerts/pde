import time
import cahn_hilliard as ch
import allen_cahn as ac
import cahn_hilliard_BDF2 as ch2
from plot import plot2d, plot3d, animation, plot_errors, plot_energies, plot_combined_energies, plot_modified_energies
import matplotlib.pyplot as plt



iterations, errors, energies, modified_energies = ch2.cahn_hilliard(
        type_of_linearisation="newton",              
        nb_of_spatial_steps=50,                                   
        nb_of_time_steps=12,   
        final_time=0.06, 
        eps=0.07
    )

plot_errors(iterations, errors, [1,2,3,4,5,6,7,8,9,10])

plot_energies(iterations, energies, [1,2,3,4,5,6,7,8,9,10])

plot_modified_energies(iterations, modified_energies,[1,2,3,4,5,6,7,8,9,10])

animation()