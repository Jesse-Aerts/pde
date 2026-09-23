import time
import cahn_hilliard as ch
import allen_cahn as ac
import cahn_hilliard_BDF2 as ch2
from plot import plot2d, plot3d, animation, plot_errors, plot_energies, plot_combined_energies, plot_modified_energies
import matplotlib.pyplot as plt



iterations, errors, energies, modified_energies = ch2.cahn_hilliard(
        type_of_linearisation="L",              
        nb_of_spatial_steps=50,                                   
        nb_of_time_steps=10,   
        final_time=1, 
        eps=0.07
    )

plot_errors(iterations, errors, [6,7,8,9,10,11,12,13,14,15])

plot_energies(iterations, energies, [6,7,8,9,10,11,12,13,14,15])

plot_modified_energies(iterations, modified_energies,[6,7,8,9,10,11,12,13,14,15])

animation()