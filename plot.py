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

def plot_errors():
    # 1. Load the data from the file
    errors_path = os.path.join("outputs", "errors.txt")

    with open(errors_path, "r") as f:
        # Read each line, strip whitespace, and convert to a float
        errors = [float(line.strip()) for line in f]

    # 2. Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(errors, marker='o', color='crimson', linestyle='-', linewidth=1.5, label='Error')

    # 3. Apply the logarithmic scale to the y-axis
    plt.yscale('log')

    # 4. Add labels, grid, and title
    plt.xlabel('Iteration / Step')
    plt.ylabel('Error Value (Log Scale)')
    plt.title('Error Convergence over Time')

    # "which='both'" ensures gridlines show for both major and minor log intervals
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    # 5. Save or show the plot
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "error_plot.png"), dpi=300)
    plt.show()

def plot_energies():
    energies_path = os.path.join("outputs", "energies.txt")

    with open(energies_path, "r") as f:
        # Read each line, strip whitespace, and convert to a float
        energies = [float(line.strip()) for line in f]

    # 2. Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(energies, marker='o', color='teal', linestyle='-', linewidth=1.5, label='Energy')

    # 3. Add labels, grid, and title (defaults to linear scale)
    plt.xlabel('Iteration / Step')
    plt.ylabel('Energy Value')
    plt.title('Energy over Time')

    # Simple grid is perfect for linear scales
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # 4. Save or show the plot
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "energy_plot.png"), dpi=300)
    plt.show()


def plot_errors2(num_timesteps_to_plot=5):
    """
    Plots the error convergence for a specified number of timesteps.
    
    Parameters:
    -----------
    num_timesteps_to_plot : int
        The number of timesteps you want to plot (e.g., plot the first 5 timesteps).
    """
    # 1. Load the errors and the iterations per timestep
    errors_path = os.path.join("outputs", "errors.txt")
    iterations_path = os.path.join("outputs", "iterations.txt")

    # Read errors
    with open(errors_path, "r") as f:
        errors = [float(line.strip()) for line in f]

    # Read iterations per timestep (assumed to be one integer per line)
    with open(iterations_path, "r") as f:
        iterations_per_timestep = [int(line.strip()) for line in f]

    # 2. Slice errors into their respective timesteps
    timesteps_errors = []
    current_index = 0
    
    for num_iters in iterations_per_timestep:
        # Extract the errors belonging to this specific timestep
        timestep_slice = errors[current_index : current_index + num_iters]
        timesteps_errors.append(timestep_slice)
        current_index += num_iters

    # 3. Setup the plot
    plt.figure(figsize=(9, 6))

    # Determine how many timesteps we can actually plot
    limit = min(num_timesteps_to_plot, len(timesteps_errors))

    # 4. Plot each timestep's error trajectory
    for t in range(limit):
        # x-axis is local to each timestep (1 to N iterations)
        x_vals = list(range(1, len(timesteps_errors[t]) + 1))
        plt.plot(
            x_vals, 
            timesteps_errors[t], 
            marker='o', 
            linestyle='-', 
            linewidth=1.5, 
            label=f'Timestep {t+1}'
        )

    # 5. Apply the logarithmic scale and labels
    plt.yscale('log')
    plt.xlabel('Iteration within Timestep')
    plt.ylabel('Error Value (Log Scale)')
    plt.title(f'Error Convergence for First {limit} Timestep(s)')

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    # 6. Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "error_plot_per_timestep.png"), dpi=300)
    plt.show()


def plot_energies2(num_timesteps_to_plot=5):
    """
    Plots the error convergence for a specified number of timesteps.
    
    Parameters:
    -----------
    num_timesteps_to_plot : int
        The number of timesteps you want to plot (e.g., plot the first 5 timesteps).
    """
    # 1. Load the errors and the iterations per timestep
    errors_path = os.path.join("outputs", "energies.txt")
    iterations_path = os.path.join("outputs", "iterations.txt")

    # Read errors
    with open(errors_path, "r") as f:
        errors = [float(line.strip()) for line in f]

    # Read iterations per timestep (assumed to be one integer per line)
    with open(iterations_path, "r") as f:
        iterations_per_timestep = [int(line.strip()) for line in f]

    # 2. Slice errors into their respective timesteps
    timesteps_energies = []
    current_index = 0
    
    for num_iters in iterations_per_timestep:
        # Extract the errors belonging to this specific timestep
        timestep_slice = errors[current_index : current_index + num_iters]
        timesteps_energies.append(timestep_slice)
        current_index += num_iters

    # 3. Setup the plot
    plt.figure(figsize=(9, 6))

    # Determine how many timesteps we can actually plot
    limit = min(num_timesteps_to_plot, len(timesteps_energies))

    # 4. Plot each timestep's error trajectory
    for t in range(limit):
        # x-axis is local to each timestep (1 to N iterations)
        x_vals = list(range(1, len(timesteps_energies[t]) + 1))
        plt.plot(
            x_vals, 
            timesteps_energies[t], 
            marker='o', 
            linestyle='-', 
            linewidth=1.5, 
            label=f'Timestep {t+1}'
        )

    # 5. Apply the logarithmic scale and labels
    plt.xlabel('Iteration within Timestep')
    plt.ylabel('Error Value (Log Scale)')
    plt.title(f'Error Convergence for First {limit} Timestep(s)')

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    # 6. Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "energy_plot_per_timestep.png"), dpi=300)
    plt.show()

def plot_errors3(num_timesteps_to_plot=5):
    # 1. Load the data from the files
    errors_path = os.path.join("outputs", "errors.txt")
    iterations_path = os.path.join("outputs", "iterations.txt")

    with open(errors_path, "r") as f:
        errors = [float(line.strip()) for line in f]
        
    with open(iterations_path, "r") as f:
        iterations_per_timestep = [int(line.strip()) for line in f]

    # 2. Determine where to stop the list
    # Make sure we don't try to read more timesteps than we actually have
    limit = min(num_timesteps_to_plot, len(iterations_per_timestep))
    
    # Add up all the iterations for those specific timesteps to find the cutoff index
    total_iterations = sum(iterations_per_timestep[:limit])

    # Slice the errors list so it simply stops after the requested timesteps
    errors_to_plot = errors[:total_iterations]

    # 3. Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(errors_to_plot, marker='o', color='crimson', linestyle='-', linewidth=1.5, label='Error')

    # 4. Apply the logarithmic scale to the y-axis
    plt.yscale('log')

    # 5. Add labels, grid, and title
    plt.xlabel('Total Iterations')
    plt.ylabel('Error Value (Log Scale)')
    plt.title(f'Error Convergence (First {limit} Timesteps)')

    # "which='both'" ensures gridlines show for both major and minor log intervals
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    # 6. Save or show the plot
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "error_plot_truncated.png"), dpi=300)
    plt.show()

def plot_errors4(timesteps=-1):
    """
    Plots error convergence as a single, continuous line for specified timesteps.
    
    Parameters:
    -----------
    timesteps : int, list, or -1
        - If int (e.g. 3): Plots only timestep 3 as a single line.
        - If list (e.g. [1, 2, 5]): Chains timesteps 1, 2, and 5 together into one continuous line.
        - If -1: Plots only the very last timestep.
    """
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

def plot_energies4(timesteps=-1):
    """
    Plots energy values as a single, continuous line for specified timesteps.
    
    Parameters:
    -----------
    timesteps : int, list, or -1
        - If int (e.g. 3): Plots only timestep 3 as a single line.
        - If list (e.g. [1, 2, 5]): Chains timesteps 1, 2, and 5 together into one continuous line.
        - If -1: Plots only the very last timestep.
    """
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
    plt.ylabel('Energy Value')
    
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