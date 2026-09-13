import os
import pyvista as pv
import matplotlib.pyplot as plt

def get_unique_filename(directory, base_name, extension):
    """
    Genereert een unieke bestandsnaam in de opgegeven map.
    Als 'plot2d.png' al bestaat, wordt dit 'plot2d_1.png', 'plot2d_2.png', enz.
    """
    os.makedirs(directory, exist_ok=True)
    filename = f"{base_name}.{extension}"
    filepath = os.path.join(directory, filename)
    
    counter = 1
    while os.path.exists(filepath):
        filename = f"{base_name}_{counter}.{extension}"
        filepath = os.path.join(directory, filename)
        counter += 1
        
    return filepath

def plot2d(time_step):
    reader = pv.get_reader("graphs/output_solution.pvd")
    reader.set_active_time_value(reader.time_values[time_step])
    grid = reader.read()[0]
    
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="c", cmap="coolwarm", clim=[-1, 1], scalar_bar_args={'title': 'u'})

    # Sla een unieke afbeelding op
    save_path = get_unique_filename("outputs", f"plot2d_step{time_step}", "png")
    plotter.show(auto_close=False)
    plotter.screenshot(save_path)
    plotter.close()
    print(f"Plot opgeslagen als: {save_path}")

def plot3d(time_step):
    reader = pv.get_reader("graphs/output_solution.pvd")
    reader.set_active_time_value(reader.time_values[time_step])
    grid = reader.read()[0]

    warp_factor = 0.1 
    grid_3d = grid.warp_by_scalar(scalars="c", factor=warp_factor)

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid_3d, 
        scalars="c", 
        cmap="coolwarm", 
        clim=[-1, 1],
        show_edges=True,
        scalar_bar_args={'title': 'u'}
    )
    plotter.camera_position = 'iso'

    # Sla een unieke afbeelding op
    save_path = get_unique_filename("outputs", f"plot3d_step{time_step}", "png")
    plotter.show(auto_close=False)
    plotter.screenshot(save_path)
    plotter.close()
    print(f"Plot opgeslagen als: {save_path}")

def animation():
    reader = pv.get_reader("graphs/output_solution.pvd")
    plotter = pv.Plotter(off_screen=True)

    reader.set_active_time_value(reader.time_values[0])
    grid = reader.read()[0]
    plotter.add_mesh(grid, scalars="c", cmap="coolwarm", clim=[-1, 1])

    save_path = get_unique_filename("simulations", "simulation", "mp4")
    plotter.open_movie(save_path, framerate=10)

    for time in reader.time_values:
        reader.set_active_time_value(time)
        new_grid = reader.read()[0]
        grid["c"] = new_grid["c"]
        plotter.write_frame()

    plotter.close()
    print(f"Animatie opgeslagen als: {save_path}")

def plot_errors(iterations_per_timestep, errors, timesteps=-1):
    if iterations_per_timestep is False or errors is False:
        errors_path = os.path.join("outputs", "errors.txt")
        iterations_path = os.path.join("outputs", "iterations.txt")

        with open(errors_path, "r") as f:
            errors = [float(line.strip()) for line in f]
            
        with open(iterations_path, "r") as f:
            iterations_per_timestep = [int(line.strip()) for line in f]

    total_timesteps = len(iterations_per_timestep)

    timestep_ranges = []
    current_index = 0
    for num_iters in iterations_per_timestep:
        timestep_ranges.append((current_index, current_index + num_iters))
        current_index += num_iters

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

    errors_to_plot = []
    for t in selected_timesteps:
        start_idx, end_idx = timestep_ranges[t - 1]
        errors_to_plot.extend(errors[start_idx:end_idx])

    plt.figure(figsize=(8, 5))
    plt.plot(
        errors_to_plot, 
        marker='o', 
        color='crimson', 
        linestyle='-', 
        linewidth=1.5, 
        label='Error'
    )

    plt.yscale('log')
    plt.xlabel('Combined Iterations')
    plt.ylabel('Error Value (Log Scale)')

    """
    if timesteps == -1:
        plt.title('Error Convergence (Last Timestep)')
    elif isinstance(timesteps, int):
        plt.title(f'Error Convergence (Timestep {timesteps})')
    else:
        plt.title(f'Error Convergence (Timesteps: {selected_timesteps})')
    """

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Sla op met unieke bestandsnaam
    save_path = get_unique_filename("outputs", "error_plot", "png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Error plot opgeslagen als: {save_path}")

def plot_energies(iterations_per_timestep, energies, timesteps=-1):
    if iterations_per_timestep is False or energies is False:
        energies_path = os.path.join("outputs", "energies.txt")
        iterations_path = os.path.join("outputs", "iterations.txt")

        with open(energies_path, "r") as f:
            energies = [float(line.strip()) for line in f]
            
        with open(iterations_path, "r") as f:
            iterations_per_timestep = [int(line.strip()) for line in f]

    total_timesteps = len(iterations_per_timestep)

    timestep_ranges = []
    current_index = 0
    for num_iters in iterations_per_timestep:
        timestep_ranges.append((current_index, current_index + num_iters))
        current_index += num_iters

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

    energies_to_plot = []
    for t in selected_timesteps:
        start_idx, end_idx = timestep_ranges[t - 1]
        energies_to_plot.extend(energies[start_idx:end_idx])

    plt.figure(figsize=(8, 5))
    plt.plot(
        energies_to_plot, 
        marker='o', 
        color='teal', 
        linestyle='-', 
        linewidth=1.5, 
        label='Energy'
    )

    plt.xlabel('Combined Iterations')
    plt.ylabel('Ginzburg-Landau Energy')

    """
    if timesteps == -1:
        plt.title('Energy over Time (Last Timestep)')
    elif isinstance(timesteps, int):
        plt.title(f'Energy over Time (Timestep {timesteps})')
    else:
        plt.title(f'Energy over Time (Timesteps: {selected_timesteps})')
    """

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = get_unique_filename("outputs", "energy_plot", "png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Energy plot opgeslagen als: {save_path}")

def plot_modified_energies(iterations_per_timestep, energies, timesteps=-1):
    if iterations_per_timestep is False or energies is False:
        energies_path = os.path.join("outputs", "modified_energies.txt")
        iterations_path = os.path.join("outputs", "iterations.txt")

        with open(energies_path, "r") as f:
            energies = [float(line.strip()) for line in f]
            
        with open(iterations_path, "r") as f:
            iterations_per_timestep = [int(line.strip()) for line in f]

    total_timesteps = len(iterations_per_timestep)

    timestep_ranges = []
    current_index = 0
    for num_iters in iterations_per_timestep:
        timestep_ranges.append((current_index, current_index + num_iters))
        current_index += num_iters

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

    energies_to_plot = []
    for t in selected_timesteps:
        start_idx, end_idx = timestep_ranges[t - 1]
        energies_to_plot.extend(energies[start_idx:end_idx])

    plt.figure(figsize=(8, 5))
    plt.plot(
        energies_to_plot, 
        marker='o', 
        color='teal', 
        linestyle='-', 
        linewidth=1.5, 
        label='Energy'
    )

    plt.xlabel('Combined Iterations')
    plt.ylabel('Modified Energy')

    """
    if timesteps == -1:
        plt.title('Energy over Time (Last Timestep)')
    elif isinstance(timesteps, int):
        plt.title(f'Energy over Time (Timestep {timesteps})')
    else:
        plt.title(f'Energy over Time (Timesteps: {selected_timesteps})')
    """

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = get_unique_filename("outputs", "modified_energy_plot", "png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Modified energy plot opgeslagen als: {save_path}")

def plot_combined_energies(iterations_per_timestep=False, energies=False, modified_energies=False, timesteps=-1):
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

    timestep_ranges = []
    current_index = 0
    for num_iters in iterations_per_timestep:
        timestep_ranges.append((current_index, current_index + num_iters))
        current_index += num_iters

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

    energies_to_plot = []
    mod_energies_to_plot = []

    for t in selected_timesteps:
        start_idx, end_idx = timestep_ranges[t - 1]
        energies_to_plot.extend(energies[start_idx:end_idx])
        mod_energies_to_plot.extend(modified_energies[start_idx:end_idx])

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
    plt.tight_layout()

    save_path = get_unique_filename("outputs", "combined_energies_plot", "png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Combined energy plot opgeslagen als: {save_path}")