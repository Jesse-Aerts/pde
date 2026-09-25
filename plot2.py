import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def plot2d(time_step):
    reader = pv.get_reader("graphs/output_solution.pvd")
    reader.set_active_time_value(reader.time_values[time_step])
    grid = reader.read()[0]
    
    plotter = pv.Plotter()
    plotter.add_text(f"Timestep {time_step}", font_size=12)
    plotter.add_mesh(grid, scalars="u", cmap="coolwarm", clim=[-1, 1])
    plotter.show()


def plot3d(time_step):
    reader = pv.get_reader("graphs/output_solution.pvd")
    reader.set_active_time_value(reader.time_values[time_step])
    grid = reader.read()[0]

    warp_factor = 0.1 
    grid_3d = grid.warp_by_scalar(scalars="u", factor=warp_factor)

    plotter = pv.Plotter()
    plotter.add_text(f"Time Step {time_step} (3D Warp)", font_size=12)
    plotter.add_mesh(grid_3d, scalars="u", cmap="coolwarm", clim=[-1, 1], show_edges=True)
    plotter.camera_position = 'iso'
    plotter.show()


def animation():
    reader = pv.get_reader("graphs/output_solution.pvd")
    os.makedirs("simulations", exist_ok=True)

    plotter = pv.Plotter(off_screen=True)
    reader.set_active_time_value(reader.time_values[0])
    grid = reader.read()[0]
    plotter.add_mesh(grid, scalars="u", cmap="coolwarm", clim=[-5, 5])

    plotter.open_movie("simulations/simulation.mp4", framerate=10)

    for time in reader.time_values:
        reader.set_active_time_value(time)
        new_grid = reader.read()[0]
        grid["u"] = new_grid["u"]
        plotter.write_frame()

    plotter.close()


def _process_data_and_timesteps(iterations_list, data_list, timesteps):
    """Helper om data-runs te filteren op gewenste tijdsstappen."""
    if not isinstance(iterations_list[0], list):
        iterations_list = [iterations_list]
    if not isinstance(data_list[0], list):
        data_list = [data_list]

    processed_results = []

    for iters_per_ts, values in zip(iterations_list, data_list):
        total_ts = len(iters_per_ts)
        
        if timesteps == -1:
            selected_ts = [total_ts]
        elif isinstance(timesteps, int):
            selected_ts = [timesteps]
        elif isinstance(timesteps, list):
            selected_ts = [t for t in timesteps if 1 <= t <= total_ts]
        else:
            raise TypeError("timesteps moet een int, list of -1 zijn.")

        ts_ranges = []
        curr = 0
        for n in iters_per_ts:
            ts_ranges.append((curr, curr + n))
            curr += n

        chained_data = []
        for t in selected_ts:
            start, end = ts_ranges[t - 1]
            chained_data.extend(values[start:end])

        processed_results.append(chained_data)

    return processed_results, selected_ts


def plot_errors(iterations_per_timestep, errors, timesteps=-1, labels=None):
    """
    Plot H1-norm verschilfouten ||u^i - u^{i-1}||_H1 voor 1 of meerdere methodes/epsilons.
    """
    processed_errors, selected_ts = _process_data_and_timesteps(iterations_per_timestep, errors, timesteps)

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(processed_errors))] if len(processed_errors) > 1 else [r'$H^1$-error']

    plt.figure(figsize=(8, 5))
    colors = ['crimson', 'navy', 'forestgreen', 'darkorange', 'purple']

    for i, err_data in enumerate(processed_errors):
        plt.plot(
            err_data, 
            marker='o', 
            markersize=4,
            color=colors[i % len(colors)], 
            linestyle='-', 
            linewidth=1.5, 
            label=labels[i]
        )

    plt.yscale('log')
    plt.xlabel(r'Combined Iterations ($i$)')
    plt.ylabel(r'$\|u^i - u^{i-1}\|_{H^1}$')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join("outputs", "error_plot.png"), dpi=300)
    plt.show()


def plot_quadratic_convergence(iterations_per_timestep, errors, timesteps=-1, labels=None):
    """
    Toont kwadratische convergentie door ||u^i - u^{i-1}||_H1 uit te zetten
    tegen ||u^{i-1} - u^{i-2}||_H1 op een log-log schaal.
    """
    processed_errors, _ = _process_data_and_timesteps(iterations_per_timestep, errors, timesteps)

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(processed_errors))] if len(processed_errors) > 1 else ["Newton"]

    plt.figure(figsize=(7, 6))
    colors = ['crimson', 'navy', 'forestgreen', 'darkorange']

    for idx, err_data in enumerate(processed_errors):
        err = np.array(err_data)
        
        # We hebben minstens 3 iteraties nodig voor e_{i-1} en e_i
        if len(err) < 3:
            print(f"Waarschuwing: Niet genoeg iteraties in run {idx+1} om convergentie-orde te berekenen.")
            continue

        e_prev = err[:-1]
        e_curr = err[1:]

        # Plot de numerieke data (e_i vs e_{i-1})
        plt.loglog(
            e_prev, e_curr, 
            marker='o', 
            markersize=5, 
            color=colors[idx % len(colors)], 
            linestyle='-', 
            linewidth=1.5, 
            label=labels[idx]
        )

        # Print de geschatte convergentie-orde (EOC) in de terminal
        print(f"\n--- Convergentie-Orde Check ({labels[idx]}) ---")
        for i in range(1, len(e_curr)):
            if e_prev[i] > 0 and e_prev[i-1] > 0:
                p_i = np.log(e_curr[i] / e_curr[i-1]) / np.log(e_prev[i] / e_prev[i-1])
                print(f"Iteratie {i+2}: p = {p_i:.2f}")

    # Referentielijnen toevoegen
    # Bepaal het bereik van e_prev van alle data voor een mooie schaal
    all_e_prev = np.concatenate([np.array(e)[:-1] for e in processed_errors if len(e) >= 3])
    x_ref = np.geomspace(min(all_e_prev), max(all_e_prev), 100)
    
    # Ankerpunt op het eerste data-element voor goede uitlijning
    c_ref = processed_errors[0][1] / (processed_errors[0][0]**2)
    c_lin = processed_errors[0][1] / (processed_errors[0][0])
    
    plt.loglog(x_ref, c_lin * x_ref, 'k--', alpha=0.5, label='Helling = 1 (Lineair)')
    plt.loglog(x_ref, c_ref * (x_ref**2), 'k:', alpha=0.8, label='Helling = 2 (Kwadratisch)')

    plt.xlabel(r'$\|u^{i-1} - u^{i-2}\|_{H^1}$')
    plt.ylabel(r'$\|u^i - u^{i-1}\|_{H^1}$')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join("outputs", "quadratic_convergence_check.png"), dpi=300)
    plt.show()


def plot_energies(iterations_per_timestep, energies, timesteps=-1):
    """
    Plot Ginzburg-Landau energie E(u) voor een enkele run.
    """
    processed_energies, selected_ts = _process_data_and_timesteps(iterations_per_timestep, energies, timesteps)
    energy_data = processed_energies[0]

    plt.figure(figsize=(8, 5))
    plt.plot(
        energy_data, 
        marker='o', 
        markersize=4,
        color='teal', 
        linestyle='-', 
        linewidth=1.5, 
        label=r'$E(u)$'
    )

    plt.xlabel(r'Combined Iterations ($i$)')
    plt.ylabel(r'Ginzburg-Landau Energy')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join("outputs", "energy_plot.png"), dpi=300)
    plt.show()


def plot_modified_energies(iterations_per_timestep, energies, timesteps=-1):
    """
    Plot aangepaste functionaal J(u) voor een enkele run.
    """
    processed_energies, selected_ts = _process_data_and_timesteps(iterations_per_timestep, energies, timesteps)
    energy_data = processed_energies[0]

    plt.figure(figsize=(8, 5))
    plt.plot(
        energy_data, 
        marker='s', 
        markersize=4,
        color='darkgreen', 
        linestyle='--', 
        linewidth=1.5, 
        label=r'$J(u)$'
    )

    plt.xlabel(r'Combined Iterations ($i$)')
    plt.ylabel(r'Modified Energy')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join("outputs", "modified_functional_plot.png"), dpi=300)
    plt.show()


def plot_combined_energies(iterations_per_timestep, energies, modified_energies, timesteps=-1):
    """
    Vergelijk E(u) en J(u) op één plot voor een enkele run.
    """
    proc_energies, selected_ts = _process_data_and_timesteps(iterations_per_timestep, energies, timesteps)
    proc_mod_energies, _ = _process_data_and_timesteps(iterations_per_timestep, energies, timesteps)

    plt.figure(figsize=(8, 5))
    plt.yscale('log')
    
    plt.plot(
        proc_energies[0], 
        marker='o', 
        markersize=4,
        color='teal', 
        linestyle='-', 
        linewidth=1.5, 
        label=r'$E(u)$'
    )
    plt.plot(
        proc_mod_energies[0], 
        marker='s', 
        markersize=4,
        color='crimson', 
        linestyle='--', 
        linewidth=1.5, 
        label=r'$J(u)$'
    )

    plt.xlabel(r'Combined Iterations ($k$)')
    plt.ylabel(r'Energy ')
    plt.title(f'Energy & Functional Evolution')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join("outputs", "combined_energies_plot.png"), dpi=300)
    plt.show()