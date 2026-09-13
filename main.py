import time
import cahn_hilliard as ch
import cahn_hilliard_LU as chl
import cahn_hilliard_test as cht
import allen_cahn as ac
from plot import plot2d, plot3d, animation, plot_errors, plot_energies, plot_combined_energies, plot_modified_energies
import matplotlib.pyplot as plt

eps = 0.07


iterations, errors, energies, modified_energies = ch.cahn_hilliard(
        type_of_linearisation="newton",              
        nb_of_spatial_steps=50,                                   
        nb_of_time_steps=500,   
        final_time=1, 
        eps=0.07
    )

for i in range(20):
    plot2d(i)

for i in range(43):
    plot2d(i*10)

#animation()