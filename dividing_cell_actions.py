import random
import numpy as np
from constants import *
from alive_cell_actions import check_room_in_grid, check_room_in_new_positions

def random_death(x, y, state, grid, new_positions, new_states, death_probability):
    if random.random() < death_probability:
        new_states.append((DEAD, '')) # Attached the EMT state though the dead state cell will disappear in the next round of update
        new_positions.append((x, y))
        # Update the grid promptly in order to reflect the current grid status
        grid[x, y]['primary_state'] = DEAD
        grid[x, y]['emt_state'] = state[1]
        return True
    return False

# def random_senescence(x, y, state, grid, new_positions, new_states, senescence_probability):
#     if random.random() < senescence_probability:
#         new_states.append((SENESCENT, state[1]))
#         new_positions.append((x, y))  # Add the new cell position
#         # Update the grid promptly in order to reflect the current grid status
#         grid[x, y]['primary_state'] = SENESCENT
#         grid[x, y]['emt_state'] = state[1]
#         return True
#     return False

# def calculate_N_ALIVE_DIVIDING(grid):
#     num_alive = np.sum(grid['primary_state'] == ALIVE)
#     num_dividing = np.sum(grid['primary_state'] == DIVIDING)
#     sum_alive_dividing = num_alive + num_dividing
#     return sum_alive_dividing

def check_senescence(x, y, state, grid, senescent_prob, new_positions, new_states, wound_positions):
    if random.random() < senescent_prob:
        new_states.append((SENESCENT, state[1]))
        new_positions.append((x, y))
        # Update the grid promptly in order to reflect the current grid status
        grid[x, y]['primary_state'] = SENESCENT
        grid[x, y]['emt_state'] = state[1]
        if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((x, y))
        return True
    else:
        return False


def division(x, y, state, grid, new_positions, new_states, division_count, wound_positions):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    random.shuffle(neighbors)

    open_neighbors = []  # List to store valid, open neighbors

    # Collect open neighbors
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy

        # Ensure the new position is within the grid boundaries
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')) and ((nx, ny) not in new_positions):  # Check if the spot is open (empty)
                open_neighbors.append((nx, ny))
    if open_neighbors:
        new_position = random.choice(open_neighbors)

        # new_states.append((ALIVE, state[1]))
        new_states.append((ALIVE, H))
        new_positions.append(new_position)  # Add the new cell position
        # new_states.append((ALIVE, state[1]))
        new_states.append((ALIVE, H))
        new_positions.append((x, y))
        division_count += 1  # Count division
        # Update the grid promptly in order to reflect the current grid status for next cells' division and migration in a single update step
        grid[new_position[0], new_position[1]]['primary_state'] = ALIVE
        grid[new_position[0], new_position[1]]['emt_state'] = H

        # If the new cell is placed in the wound region, mark it as updated
        if (30 <= new_position[0] <= 69):
            wound_positions.add((new_position[0], new_position[1]))  # Add this position to the updated wound positions
        return True, division_count
    else:
        return False, division_count

# Define a function for keeping a cell alive
def check_alive(x, y, state, new_positions, new_states, wound_positions):
    new_states.append((ALIVE, state[1]))
    new_positions.append((x, y))  # Keep the original position
    if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
        wound_positions.add((x, y))
    return True  # Cell stays alive

# Function to choose a random action for each cell
def dividing_random_action(x, y, state, grid, new_positions, new_states, division_count, death_probability, wound_positions, age, step):
    # If the cell is senescent, it remains in its state and is not processed further
    if ((grid[x, y]["primary_state"], grid[x, y]["emt_state"]) == (SENESCENT, H)) or ((grid[x, y]["primary_state"], grid[x, y]["emt_state"]) == (SENESCENT, E)):
        new_states.append((SENESCENT, state[1]))
        new_positions.append((x, y))
        if 30 <= x <= 69:  # Mark the wound position as updated if applicable
            wound_positions.add((x, y))
        return division_count
    
    if check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        actions = [
            lambda: (random_death(x, y, state, grid, new_positions, new_states, death_probability), division_count),
            lambda: division(x, y, state, grid, new_positions, new_states, division_count, wound_positions),
        ]
    else:
        actions = [
            lambda: (random_death(x, y, state, grid, new_positions, new_states, death_probability), division_count),
            lambda: (check_alive(x, y, state, new_positions, new_states, wound_positions), division_count)
        ]

    # N_ALIVE_DIVIDING = calculate_N_ALIVE_DIVIDING(grid)
    # senescent_prob = ((senescence_rate)*(age+step))/N_ALIVE_DIVIDING
    senescent_prob = senescence_rate*(age+step)

    if check_senescence(x, y, state, grid, senescent_prob, new_positions, new_states, wound_positions):
        return division_count
    else:
        random.shuffle(actions)
        for action in actions:
            success, division_count = action()
            if success:
                break

    return division_count  # Return the updated migration_count
