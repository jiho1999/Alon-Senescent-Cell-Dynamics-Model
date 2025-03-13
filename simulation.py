# simulation.py

import numpy as np
import random
from constants import *
from initialization import initialize_grid, initialize_cells
from alive_cell_actions import all_neighbors_empty, all_neighbors_occupied, alive_random_action
from dividing_cell_actions import dividing_random_action
from senescent_cell_actions import senescent_action
from utils import *
import pandas as pd

def run_simulation(age, num_steps, runs=100):
    for run in range(runs):
        grid = initialize_grid(grid_size_x, grid_size_y)
        cell_positions, cell_states = initialize_cells(grid_size_x, grid_size_x)
        
        results = []
        division_counts = []
        migration_counts = []
        avg_permeability_lst = []
        wound_empty_dead_counts = []  # List to store the count of EMPTY/DEAD in wound area per step
        wound_positions = set() # Variable to track when all wound area is update
        wound_closed_step = None # To record the step when all wound positions are updated
        wound_area = set((x, y) for x in range(30, 70) for y in range(grid_size_y)) # Define the full set of wound positions (x = 30 to x = 69 across all y)
        senescent_number = []
        
        for step in range(num_steps):
            # Visualize the initial grid of alive and wound area (0-29 and 70-99: alive, 30-69: wound)
            if step == 0:
                grid, color_grid = update_grid(grid, cell_positions, cell_states, grid_size_x, grid_size_y)
                print(grid.dtype)
                visualize_grid(color_grid, step, run, age, save_images=True)
                print(step)
                count_1 = np.count_nonzero(grid['primary_state'] == 1)
                count_2 = np.count_nonzero(grid['primary_state'] == 2)
    
                print(f"Number of grid elements that are 1: {count_1}")
                print(f"Number of grid elements that are 2: {count_2}")

            # Process cell actions and update grid, cell positions, and cell states here
            new_positions, new_states = [], []
            migration_count = 0
            division_count = 0

            indices = list(range(cell_positions.shape[0]))
            random.shuffle(indices)

            for i in indices:
                x, y = cell_positions[i]
                state = cell_states[i]
                
                # Update the EMT cell type
                if (state[1] == E) and not(all_neighbors_occupied(x, y, grid, grid_size_x, grid_size_y, EMPTY)):
                    state[0] = ALIVE
                    state[1] = H
                    
                if (state[1] == E or state[1] == H) and (state[0] != SENESCENT) and all_neighbors_empty(x, y, grid, grid_size_x, grid_size_y, EMPTY):
                    state[0] = ALIVE
                    state[1] = M
                
                if (state[0] == SENESCENT) and (state[1] == (E or H) and all_neighbors_empty(x, y, grid, grid_size_x, grid_size_y, EMPTY)):
                    state[0] = SENESCENT
                    state[1] = state[1]

                if (state[1] == H) and all_neighbors_occupied(x, y, grid, grid_size_x, grid_size_y, EMPTY):
                    state[1] = E

                if (state[1] == M) and all_neighbors_occupied(x, y, grid, grid_size_x, grid_size_y, EMPTY):
                    if M_to_E_probability:
                        state[1] = E
                
                #if (state[0] == DIVIDING and state[1] == E) and not (all_neighbors_occupied(x, y, grid, grid_size_x, grid_size_y, EMPTY)):

                if int(state[0]) == ALIVE:
                    migration_count = alive_random_action(x, y, state, grid, new_positions, new_states, migration_count, division_probability, death_probability, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions, step, age)

                elif int(state[0]) == DIVIDING:
                    # If there's an open spot, divide the cell and place the new cell
                    if not all_neighbors_occupied(x, y, grid, grid_size_x, grid_size_y, EMPTY):
                        division_count = dividing_random_action(x, y, state, grid, new_positions, new_states, division_count, death_probability, wound_positions, age, step)
                    else:
                        new_states.append((ALIVE, state[1]))
                        new_positions.append((x, y))

                elif int(state[0]) == DEAD:
                    new_states.append((EMPTY, ''))
                    new_positions.append((x, y))

                    # Update the grid promptly in order to reflect the current grid status for next cells' division and migration in a single update step
                    grid[x, y]['primary_state'] = EMPTY
                    grid[x, y]['emt_state'] = ''
                    # Dead cells are not added to new_states or new_positions after this cycle
                    continue  # Skip adding this cell to the new lists
                
                elif int(state[0]) == SENESCENT:
                    migration_count = senescent_action(x, y, state, grid, new_positions, new_states, migration_count, removal_rate_ß, wound_positions)
                    continue # Skip further processing for this cell
            
            # After processing all cells for this step, check if the wound area is fully updated
            if wound_area == wound_positions and wound_closed_step is None:
                wound_closed_step = step + 1
                print(f"All wound positions were updated at step {wound_closed_step}")
            
            # Store the division and migration count for each step of update
            division_counts.append(division_count)
            migration_counts.append(migration_count)

            # Update positions and states
            cell_positions, cell_states = np.array(new_positions), np.array(new_states)

            # Visualization (update_grid will update the grid and return the color grid to visualize using visualize_grid)
            grid, color_grid = update_grid(grid, cell_positions, cell_states, grid_size_x, grid_size_y)
            visualize_grid(color_grid, step + 1, run, age, save_images=True)
            print(step)

            # Calculate and append the permeability for each step
            avg_permeability_lst.append(calculate_permeability(grid))

            # Count EMPTY or DEAD cells in the wound area for this step
            empty_dead_count = sum(1 for (x, y) in wound_area if (grid[x, y]['primary_state'], grid[x, y]['emt_state']) in {(EMPTY, ''), (DEAD, '')})
            wound_empty_dead_counts.append(empty_dead_count)

            # Count the number of senescent cell
            senescent_count = sum(1 for state in new_states if state[0] == SENESCENT)
            senescent_number.append(senescent_count)

        # Save data
        for step in range(num_steps):
            results.append([age, step + 1, division_counts[step], migration_counts[step], avg_permeability_lst[step], wound_empty_dead_counts[step], senescent_number[step]])
        print(age)
        
        filename = f'division_migration_senescence_{age:.1e}_run_{run + 1}.xlsx'
        df_results = pd.DataFrame(results, columns=['Age(h)', 'Step(h)', 'Division Count', 'Migration Count', 'Average Permeability', 'Wound Area', 'Senescent Number'])
        df_results['Wound Closure Step'] = wound_closed_step if wound_closed_step is not None else 'Not closed yet'
        df_results.to_excel(filename, index=False)

        # # Plot the data
        # plot_results(filename)
