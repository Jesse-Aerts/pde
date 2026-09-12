import pyvista as pv
import os
import matplotlib.pyplot as plt

def plot2d(time_step):
    # Load the PVD file
    reader = pv.get_reader("graphs/output_solution.pvd")

    # Last step (index -1)
    reader.set_active_time_value(reader.time_values[time_step])
    grid = reader.read()[0]
    plotter = pv.Plotter()

    plotter.add_text("Timestep", font_size=12)
    plotter.add_mesh(grid, scalars="c", cmap="coolwarm", clim=[-1, 1])

    plotter.link_views()  # Syncs camera movement between both views!
    plotter.show()

def plot3d(time_step):
    # Load the PVD file
    reader = pv.get_reader("graphs/output_solution.pvd")

    # 1. Read the first and last time steps
    # First step
    reader.set_active_time_value(reader.time_values[time_step])
    grid = reader.read()[0]

  
    # 2. Warp the 2D grid into 3D based on the value of 'c'
    # 'factor' controls how high the 3D peaks will be
    warp_factor = 0.1 
    grid_3d = grid.warp_by_scalar(scalars="c", factor=warp_factor)
    

    # 3. Setup a 1x2 subplot window for 3D viewing
    plotter = pv.Plotter()

    plotter.add_text("First Time Step (3D)", font_size=12)
    plotter.add_mesh(
        grid_3d, 
        scalars="c", 
        cmap="coolwarm", 
        clim=[-1, 1],
        show_edges=True  # Optional: displays the mesh grid lines in 3D
    )

    # Set a nice starting 3D isometric camera angle
    plotter.camera_position = 'iso'

    # Sync mouse rotations between both views
    plotter.link_views()  
    plotter.show()

def animation():
    reader = pv.get_reader("graphs/output_solution.pvd")

    # Use off_screen=True so it renders silently in the background
    plotter = pv.Plotter(off_screen=True)

    # Read initial frame
    reader.set_active_time_value(reader.time_values[0])
    grid = reader.read()[0]
    mesh_actor = plotter.add_mesh(grid, scalars="c", cmap="coolwarm", clim=[-1, 1])

    # 1. Open the MP4 file (Note: keyword is framerate, NOT fps)
    plotter.open_movie("simulations/simulation.mp4", framerate=10)

    # 2. Loop and record frames
    for time in reader.time_values:
        reader.set_active_time_value(time)
        new_grid = reader.read()[0]
        
        # Update grid values in-place
        grid["c"] = new_grid["c"]
        
        # Render and capture frame
        plotter.write_frame()

    # 3. Finalize and write the MP4 file
    plotter.close()


def plot_errors(iterations_per_timestep, errors, timesteps=-1):
    """
    Plots error convergence as a single, continuous line for specified timesteps.
    
    Parameters:
    -----------
    timesteps : int, list, or -1
        - If int (e.g. 3): Plots only timestep 3 as a single line.
        - If list (e.g. [1, 2, 5]): Chains timesteps 1, 2, and 5 together into one continuous line.
        - If -1: Plots only the very last timestep.
    """

    if iterations_per_timestep == False or errors == False:
        # 1. Load the data
        errors_path = os.path.join("outputs", "errors.txt")
        iterations_path = os.path.join("outputs", "iterations.txt")

        with open(errors_path, "r") as f:
            errors = [float(line.strip()) for line in f]
            
        with open(iterations_path, "r") as f:
            iterations_per_timestep = [int(line.strip()) for line in f]

    total_timesteps = len(iterations_per_timestep)

    # 2. Map the start and end indices for every timestep
    timestep_ranges = []
    current_index = 0
    for num_iters in iterations_per_timestep:
        timestep_ranges.append((current_index, current_index + num_iters))
        current_index += num_iters

    # 3. Resolve which timesteps we want to collect
    selected_timesteps = []

    if timesteps == -1:
        selected_timesteps = [total_timesteps]
    elif isinstance(timesteps, int):
        if 1 <= timesteps <= total_timesteps:
            selected_timesteps = [timesteps]
        else:
            raise ValueError(f"Timestep {timesteps} is out of bounds (1 to {total_timesteps}).")
    elif isinstance(timesteps, list):
        for t in timesteps:
            if 1 <= t <= total_timesteps:
                selected_timesteps.append(t)
            else:
                print(f"Warning: Timestep {t} is out of bounds and will be skipped.")
    else:
        raise TypeError("timesteps must be an integer, a list of integers, or -1.")

    # 4. Chain the selected errors together into a single list
    errors_to_plot = []
    for t in selected_timesteps:
        start_idx, end_idx = timestep_ranges[t - 1]
        errors_to_plot.extend(errors[start_idx:end_idx])

    # 5. Create the single-line plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        errors_to_plot, 
        marker='o', 
        color='crimson', 
        linestyle='-', 
        linewidth=1.5, 
        label='Error'
    )

    # 6. Apply formatting
    plt.yscale('log')
    plt.xlabel('Combined Iterations')
    plt.ylabel('Error Value (Log Scale)')
    
    # Customize title based on selection
    if timesteps == -1:
        plt.title('Error Convergence (Last Timestep)')
    elif isinstance(timesteps, int):
        plt.title(f'Error Convergence (Timestep {timesteps})')
    else:
        plt.title(f'Error Convergence (Timesteps: {selected_timesteps})')

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    # 7. Save and show
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "error_plot_single_line.png"), dpi=300)
    plt.show()

def plot_energies(iterations_per_timestep, energies, timesteps=-1):
    """
    Plots energy values as a single, continuous line for specified timesteps.
    
    Parameters:
    -----------
    timesteps : int, list, or -1
        - If int (e.g. 3): Plots only timestep 3 as a single line.
        - If list (e.g. [1, 2, 5]): Chains timesteps 1, 2, and 5 together into one continuous line.
        - If -1: Plots only the very last timestep.
    """
    if iterations_per_timestep == False or energies == False:
        # 1. Load the data
        energies_path = os.path.join("outputs", "energies.txt")
        iterations_path = os.path.join("outputs", "iterations.txt")

        with open(energies_path, "r") as f:
            energies = [float(line.strip()) for line in f]
            
        with open(iterations_path, "r") as f:
            iterations_per_timestep = [int(line.strip()) for line in f]

    total_timesteps = len(iterations_per_timestep)

    # 2. Map the start and end indices for every timestep
    timestep_ranges = []
    current_index = 0
    for num_iters in iterations_per_timestep:
        timestep_ranges.append((current_index, current_index + num_iters))
        current_index += num_iters

    # 3. Resolve which timesteps we want to collect
    selected_timesteps = []

    if timesteps == -1:
        selected_timesteps = [total_timesteps]
    elif isinstance(timesteps, int):
        if 1 <= timesteps <= total_timesteps:
            selected_timesteps = [timesteps]
        else:
            raise ValueError(f"Timestep {timesteps} is out of bounds (1 to {total_timesteps}).")
    elif isinstance(timesteps, list):
        for t in timesteps:
            if 1 <= t <= total_timesteps:
                selected_timesteps.append(t)
            else:
                print(f"Warning: Timestep {t} is out of bounds and will be skipped.")
    else:
        raise TypeError("timesteps must be an integer, a list of integers, or -1.")

    # 4. Chain the selected energies together into a single list
    energies_to_plot = []
    for t in selected_timesteps:
        start_idx, end_idx = timestep_ranges[t - 1]
        energies_to_plot.extend(energies[start_idx:end_idx])

    # 5. Create the single-line plot (linear scale)
    plt.figure(figsize=(8, 5))
    plt.plot(
        energies_to_plot, 
        marker='o', 
        color='teal', 
        linestyle='-', 
        linewidth=1.5, 
        label='Energy'
    )

    # 6. Apply formatting (linear y-axis)
    plt.xlabel('Combined Iterations')
    plt.ylabel('Ginzburg-Landau Energy')
    
    # Customize title based on selection
    if timesteps == -1:
        plt.title('Energy over Time (Last Timestep)')
    elif isinstance(timesteps, int):
        plt.title(f'Energy over Time (Timestep {timesteps})')
    else:
        plt.title(f'Energy over Time (Timesteps: {selected_timesteps})')

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # 7. Save and show
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "energy_plot_single_line.png"), dpi=300)
    plt.show()

def plot_modified_energies(iterations_per_timestep, energies, timesteps=-1):
    """
    Plots energy values as a single, continuous line for specified timesteps.
    
    Parameters:
    -----------
    timesteps : int, list, or -1
        - If int (e.g. 3): Plots only timestep 3 as a single line.
        - If list (e.g. [1, 2, 5]): Chains timesteps 1, 2, and 5 together into one continuous line.
        - If -1: Plots only the very last timestep.
    """
    if iterations_per_timestep == False or energies == False:
        # 1. Load the data
        energies_path = os.path.join("outputs", "modified_energies.txt")
        iterations_path = os.path.join("outputs", "iterations.txt")

        with open(energies_path, "r") as f:
            energies = [float(line.strip()) for line in f]
            
        with open(iterations_path, "r") as f:
            iterations_per_timestep = [int(line.strip()) for line in f]

    total_timesteps = len(iterations_per_timestep)

    # 2. Map the start and end indices for every timestep
    timestep_ranges = []
    current_index = 0
    for num_iters in iterations_per_timestep:
        timestep_ranges.append((current_index, current_index + num_iters))
        current_index += num_iters

    # 3. Resolve which timesteps we want to collect
    selected_timesteps = []

    if timesteps == -1:
        selected_timesteps = [total_timesteps]
    elif isinstance(timesteps, int):
        if 1 <= timesteps <= total_timesteps:
            selected_timesteps = [timesteps]
        else:
            raise ValueError(f"Timestep {timesteps} is out of bounds (1 to {total_timesteps}).")
    elif isinstance(timesteps, list):
        for t in timesteps:
            if 1 <= t <= total_timesteps:
                selected_timesteps.append(t)
            else:
                print(f"Warning: Timestep {t} is out of bounds and will be skipped.")
    else:
        raise TypeError("timesteps must be an integer, a list of integers, or -1.")

    # 4. Chain the selected energies together into a single list
    energies_to_plot = []
    for t in selected_timesteps:
        start_idx, end_idx = timestep_ranges[t - 1]
        energies_to_plot.extend(energies[start_idx:end_idx])

    # 5. Create the single-line plot (linear scale)
    plt.figure(figsize=(8, 5))
    plt.plot(
        energies_to_plot, 
        marker='o', 
        color='teal', 
        linestyle='-', 
        linewidth=1.5, 
        label='Energy'
    )

    # 6. Apply formatting (linear y-axis)
    plt.xlabel('Combined Iterations')
    plt.ylabel('Modified Energy')
    
    # Customize title based on selection
    if timesteps == -1:
        plt.title('Energy over Time (Last Timestep)')
    elif isinstance(timesteps, int):
        plt.title(f'Energy over Time (Timestep {timesteps})')
    else:
        plt.title(f'Energy over Time (Timesteps: {selected_timesteps})')

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # 7. Save and show
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "energy_plot_single_line.png"), dpi=300)
    plt.show()

import os
import matplotlib.pyplot as plt

def plot_combined_energies(iterations_per_timestep=False, energies=False, modified_energies=False, timesteps=-1):
    """
    Plots standard energy and modified energy values together on a single plot for specified timesteps.
    
    Parameters:
    -----------
    iterations_per_timestep : list or False
        List of iteration counts per timestep. If False, reads from file.
    energies : list or False
        List of standard energy values. If False, reads from file.
    modified_energies : list or False
        List of modified energy values. If False, reads from file.
    timesteps : int, list, or -1
        - If int (e.g. 3): Plots only timestep 3.
        - If list (e.g. [1, 2, 5]): Chains specified timesteps together.
        - If -1: Plots only the very last timestep.
    """
    # 1. Load data if needed
    if iterations_per_timestep is False:
        iterations_path = os.path.join("outputs", "iterations.txt")
        with open(iterations_path, "r") as f:
            iterations_per_timestep = [int(line.strip()) for line in f]

    if energies is False:
        energies_path = os.path.join("outputs", "energies.txt")
        with open(energies_path, "r") as f:
            energies = [float(line.strip()) for line in f]

    if modified_energies is False:
        mod_energies_path = os.path.join("outputs", "modified_energies.txt")
        with open(mod_energies_path, "r") as f:
            modified_energies = [float(line.strip()) for line in f]

    total_timesteps = len(iterations_per_timestep)

    # 2. Map start and end indices for every timestep
    timestep_ranges = []
    current_index = 0
    for num_iters in iterations_per_timestep:
        timestep_ranges.append((current_index, current_index + num_iters))
        current_index += num_iters

    # 3. Resolve which timesteps to plot
    selected_timesteps = []
    if timesteps == -1:
        selected_timesteps = [total_timesteps]
    elif isinstance(timesteps, int):
        if 1 <= timesteps <= total_timesteps:
            selected_timesteps = [timesteps]
        else:
            raise ValueError(f"Timestep {timesteps} is out of bounds (1 to {total_timesteps}).")
    elif isinstance(timesteps, list):
        for t in timesteps:
            if 1 <= t <= total_timesteps:
                selected_timesteps.append(t)
            else:
                print(f"Warning: Timestep {t} is out of bounds and will be skipped.")
    else:
        raise TypeError("timesteps must be an integer, a list of integers, or -1.")

    # 4. Chain selected energy data together
    energies_to_plot = []
    mod_energies_to_plot = []

    for t in selected_timesteps:
        start_idx, end_idx = timestep_ranges[t - 1]
        energies_to_plot.extend(energies[start_idx:end_idx])
        mod_energies_to_plot.extend(modified_energies[start_idx:end_idx])

    # 5. Create plot
    plt.figure(figsize=(8, 5))
    plt.yscale('log')
    
    plt.plot(
        energies_to_plot, 
        marker='o', 
        color='teal', 
        linestyle='-', 
        linewidth=1.5, 
        label='Ginzburg-Landau Energy'
    )
    plt.plot(
        mod_energies_to_plot, 
        marker='s', 
        color='crimson', 
        linestyle='--', 
        linewidth=1.5, 
        label='Modified Energy'
    )

    # 6. Formatting
    plt.xlabel('Combined Iterations')
    plt.ylabel('Energy Value')
    
    if timesteps == -1:
        plt.title('Energy Evolution (Last Timestep)')
    elif isinstance(timesteps, int):
        plt.title(f'Energy Evolution (Timestep {timesteps})')
    else:
        plt.title(f'Energy Evolution (Timesteps: {selected_timesteps})')

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # 7. Save and show
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "combined_energies_plot.png"), dpi=300)
    plt.show()