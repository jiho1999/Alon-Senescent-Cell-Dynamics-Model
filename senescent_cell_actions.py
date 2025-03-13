import random
import numpy as np
from constants import SENESCENT, EMPTY, H, M, grid_size_x, grid_size_y, senescent_migration_probability
from alive_cell_actions import check_room_in_grid, check_room_in_new_positions, all_neighbors_empty_mesenchymal

def calculate_N_SENESCENT(grid):
    num_senescent = np.sum(grid['primary_state'] == SENESCENT)
    return num_senescent

def calculate_senescent_percentage(grid):
    num_existing_cells = np.sum(grid['primary_state'] != EMPTY)
    num_senescent_cell = calculate_N_SENESCENT(grid)
    senescent_percentage = num_senescent_cell / num_existing_cells
    return senescent_percentage

def check_senescent_removal(x, y, grid, removal_probability, new_positions, new_states, wound_positions):
    if random.random() < removal_probability:
        new_states.append((EMPTY, ''))
        new_positions.append((x, y))
        # Update the grid promptly in order to reflect the current grid status for next cells' division and migration in a single update step
        grid[x, y]['primary_state'] = EMPTY
        grid[x, y]['emt_state'] = ''
        if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((x, y))
        return True
    else:
        return False

def check_senescent_migration(x, y, state, grid, new_positions, new_states, migration_count, senescent_migration_probability, wound_positions):
    # H is the only senescent cells that can migrate. E cannot move and M cannot be a senescent cell.
    if (state[1] == H) and random.random() < senescent_migration_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        random.shuffle(neighbors)
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')) and not(all_neighbors_empty_mesenchymal(nx, ny, grid, grid_size_x, grid_size_y, EMPTY)):
                    new_states.append((SENESCENT, H))
                    new_positions.append((nx, ny))
                    grid[x, y]['primary_state'] = EMPTY
                    grid[x, y]['emt_state'] = ''
                    grid[nx, ny]['primary_state'] = SENESCENT
                    grid[nx, ny]['emt_state'] = H
                    if 30 <= nx <= 69:  # If the cell moves into the wound region, mark the wound position as updated
                        wound_positions.add((nx, ny))
                    migration_count += 1  # Increment migration count
                    return migration_count                            
                else:
                    grid[x, y]['primary_state'] = SENESCENT
                    grid[x, y]['emt_state'] = H
                    if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
                        wound_positions.add((x, y))
                    return migration_count
    elif (state[1] == M) and random.random() < senescent_migration_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        random.shuffle(neighbors)
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')):
                    new_states.append((SENESCENT, M))
                    new_positions.append((nx, ny))
                    grid[x, y]['primary_state'] = EMPTY
                    grid[x, y]['emt_state'] = ''
                    grid[nx, ny]['primary_state'] = SENESCENT
                    grid[nx, ny]['emt_state'] = M
                    if 30 <= nx <= 69:  # If the cell moves into the wound region, mark the wound position as updated
                        wound_positions.add((nx, ny))
                    migration_count += 1  # Increment migration count
                    return migration_count                            
                else:
                    new_states.append((SENESCENT, M))
                    new_positions.append((x, y))
                    grid[x, y]['primary_state'] = SENESCENT
                    grid[x, y]['emt_state'] = M
                    if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
                        wound_positions.add((x, y))
                    return migration_count
    else:
        grid[x, y]['primary_state'] = SENESCENT
        grid[x, y]['emt_state'] = state[1]
        if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((x, y))
        return migration_count
    
def senescent_action(x, y, state, grid, new_positions, new_states, migration_count, removal_rate_ß, wound_positions):
    senescent_percentage = calculate_senescent_percentage(grid)
    senescent_number = calculate_N_SENESCENT(grid)
    # removal_probability = (removal_rate_ß*(1/(1+senescent_percentage)))/senescent_number
    removal_probability = removal_rate_ß*(1/(1+senescent_percentage))

    if check_senescent_removal(x, y, grid, removal_probability, new_positions, new_states, wound_positions):
        return migration_count
    else:
        migration_count = check_senescent_migration(x, y, state, grid, new_positions, new_states, migration_count, senescent_migration_probability, wound_positions)
        return migration_count
