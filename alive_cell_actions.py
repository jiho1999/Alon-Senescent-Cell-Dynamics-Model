# cell_actions.py

import random
import numpy as np
from constants import E, H, M, grid_size_x, grid_size_y
from constants import EMPTY, ALIVE, DEAD, DIVIDING, SENESCENT
from constants import senescence_rate

# Function to check room for division (without periodic boundary and correct boundary checks)
def check_room_in_grid(x, y, grid):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if (grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, ''):
                return True
    return False

# Function to check if room is available in the list of new positions
def check_room_in_new_positions(x, y, new_positions, grid):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if (nx, ny) not in new_positions:
                return True
    return False

def find_nonempty_grid_element (x, y, grid):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    nonempty_grid_element = []
    for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if (grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) != (EMPTY, ''):
                    nonempty_grid_element.append((nx, ny))
    return nonempty_grid_element

# Check if neighboring grid is all empty
def all_neighbors_empty(x, y, grid, grid_size_x, grid_size_y, EMPTY):
    neighbors = [
        (x+1, y+0), (x-1, y+0), (x+0, y+1), (x+0, y-1),  # Edge neighbors
        (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)   # Corner neighbors
    ]
    
    # Iterate through each neighbor and check if they are within bounds and empty
    for nx, ny in neighbors:
        if 0 <= nx < grid_size_x and 0 <= ny < grid_size_y:  # Ensure within grid bounds
            if (grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) != (EMPTY, ''):
                return False  # If any neighbor is not EMPTY, return False
    
    return True  # All neighbors are EMPTY

# Check if neighboring grid is all empty or mesenchymal cell
def all_neighbors_empty_mesenchymal(x, y, grid, grid_size_x, grid_size_y, EMPTY):
    neighbors = [
        (x+1, y+0), (x-1, y+0), (x+0, y+1), (x+0, y-1),  # Edge neighbors
        (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)   # Corner neighbors
    ]
    
    # Iterate through each neighbor and check if they are within bounds and empty
    for nx, ny in neighbors:
        if 0 <= nx < grid_size_x and 0 <= ny < grid_size_y:  # Ensure within grid bounds
            if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) != (EMPTY, '')) and (grid[nx, ny]["emt_state"] != M):
                return False  # If any neighbor is not EMPTY, return False
    
    return True  # All neighbors are EMPTY or Mesenchymal

# Check if neighboring grid is all occupied
def all_neighbors_occupied(x, y, grid, grid_size_x, grid_size_y, EMPTY):
    neighbors = [
        (x+1, y+0), (x-1, y+0), (x+0, y+1), (x+0, y-1),  # Edge neighbors
        (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)   # Corner neighbors
    ]
    
    # Iterate through each neighbor and check if they are within bounds and occupied (not EMPTY)
    for nx, ny in neighbors:
        if 0 <= nx < grid_size_x and 0 <= ny < grid_size_y:  # Ensure within grid bounds
            if (grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, ''):
                return False  # If any neighbor is EMPTY, return False
    
    return True  # All neighbors are occupied (not EMPTY)

# Function to move cells to an available empty neighboring spot
def move_cells(x, y, new_positions, grid):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    random.shuffle(neighbors)
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')) and check_room_in_new_positions(x, y, new_positions, grid):
                return nx, ny
    return x, y # Since we use move_cells function when we know there is a open spot, code will not reach return x, y

# Function to move Hybrid cells to an avaialbe empty neighboring spot where cells can remain the contact
    # Nonzero x_dir, y_dir
    # 1. move cell with x and y
    # 2. If there is no grid element that is possible based on x and y value, move cell based on bigger absolute value of x and y
    # 3. If 2. does not work, move randomly

    # Zero x and y
    # 1. Move cell randomly based on empty grid element(s)
# Maintain the contact of neighboring cells!
def move_H_cell(x, y, grid):
    # Density dependent directional movement
    # Initialize average density of x, y
    avg_x = 0
    avg_y = 0
    # Initialize the point how much in which direction the original x, y should move
    x_dir = 0
    y_dir = 0
    # Neighboring grid elements
    neighbors = [
        (x+1, y+0), (x-1, y+0), (x+0, y+1), (x+0, y-1),  # Edge neighbors
        (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)   # Corner neighbors
    ]

    # Dictionary for calculating average density x, y
    density_cal_table = {(x-1, y-1): (-1, -1), 
                         (x-1, y): (-1, 0), 
                         (x-1, y+1): (-1, 1), 
                         (x, y+1): (0, 1), 
                         (x+1, y+1): (1, 1), 
                         (x+1, y): (1, 0), 
                         (x+1, y-1): (1, -1), 
                         (x, y-1): (0, -1) }

    # Find nonempty grid element(s)
    nonempty_grid_element = find_nonempty_grid_element (x, y, grid)
    # Emtpy grid element(s)
    empty_grid_element = list(set(neighbors) - set(nonempty_grid_element))

    # Calculate average density x, y
    for point in nonempty_grid_element:
        avg_x += density_cal_table[point][0]
        avg_y += density_cal_table[point][1]

    avg_x /= len(nonempty_grid_element)
    avg_y /= len(nonempty_grid_element)

    # Update direction, (x_dir, y_dir) 
    x_dir = 0 if avg_x == 0 else (1 if avg_x < 0 else -1)
    y_dir = 0 if avg_y == 0 else (1 if avg_y < 0 else -1)

    # Move cell using average density and empty grid
    # Nonzero x_dir or y_dir when the cell can remain contact
    if (x_dir != 0 or y_dir != 0) and (not(all_neighbors_empty(x+x_dir, y+y_dir, grid, grid_size_x, grid_size_y, EMPTY))):
        if 0 <= (x + x_dir) < grid.shape[0] and 0 <= (y + y_dir) < grid.shape[1]:
            if ((grid[x + x_dir, y + y_dir]["primary_state"], grid[x + x_dir, y + y_dir]["emt_state"]) == (EMPTY, '')):
                return x+x_dir, y+y_dir
    # Nonzero x_dir or y_dir when the cell cannot remain contact
    elif (x_dir != 0 or y_dir != 0) and (all_neighbors_empty(x+x_dir, y+y_dir, grid, grid_size_x, grid_size_y, EMPTY)):
        # Decide the direction based on the density difference of x and y
        if (avg_x > avg_y) and (not(all_neighbors_empty(x+x_dir, y+y_dir, grid, grid_size_x, grid_size_y, EMPTY))):
            if 0 <= (x + x_dir) < grid.shape[0] and 0 <= (y) < grid.shape[1]:
                if ((grid[x + x_dir, y]["primary_state"], grid[x + x_dir, y]["emt_state"]) == (EMPTY, '')):
                    return x+x_dir, y
        elif (avg_y > avg_x) and (not(all_neighbors_empty(x+x_dir, y+y_dir, grid, grid_size_x, grid_size_y, EMPTY))):
            if 0 <= (x) < grid.shape[0] and 0 <= (y + y_dir) < grid.shape[1]:
                if ((grid[x, y + y_dir]["primary_state"], grid[x, y + y_dir]["emt_state"]) == (EMPTY, '')):
                    return x, y + y_dir
        else:
            random.shuffle(empty_grid_element)
            for point in empty_grid_element:
                if 0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1]:
                    if not(all_neighbors_empty(point[0], point[1], grid, grid_size_x, grid_size_y, EMPTY)):
                        if ((grid[x + x_dir, y + y_dir]["primary_state"], grid[x + x_dir, y + y_dir]["emt_state"]) == (EMPTY, '')):
                            return point[0], point[1]
    # Zero x_dir and y_dir: randomly select the migration direction to open space remaining the contact
    elif (x_dir == 0 and y_dir == 0):
        random.shuffle(empty_grid_element)
        for point in empty_grid_element:
            if 0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1]:
                if not(all_neighbors_empty(point[0], point[1], grid, grid_size_x, grid_size_y, EMPTY)):
                    if ((grid[point[0], point[1]]["primary_state"], grid[point[0], point[1]]["emt_state"]) == (EMPTY, '')):
                        return point[0], point[1]

    return x, y # Since we use move_cells function when we know there is a open spot, code will not reach return x, y

# Define a function for cell division
def check_division(x, y, state, grid, new_positions, new_states, division_probability, wound_positions):
    if random.random() < division_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        #new_states.append((DIVIDING, state[1]))  # Enter dividing state
        new_states.append((DIVIDING, H))  # Enter dividing state
        new_positions.append((x, y))  # Keep the original cell's position
        # Update the grid promptly in order to reflect the current grid status
        grid[x, y]['primary_state'] = DIVIDING
        grid[x, y]['emt_state'] = H
        if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((x, y))
        return True  # Division occurred
    return False  # Division didn't happen

# Define a function for cell death
def check_death(x, y, state, grid, new_positions, new_states, death_probability, wound_positions):
    if random.random() < death_probability:  # Chance to die
        new_states.append((DEAD, ''))
        new_positions.append((x, y))  # Keep the dead cell in the grid for this cycle
        if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((x, y))
        # Update the grid promptly in order to reflect the current grid status
        grid[x, y]['primary_state'] = DEAD
        grid[x, y]['emt_state'] = state[1]
        return True  # Death occurred
    return False  # Death didn't happen

# Define a function for cell migration (modifies migration_count)
def check_migration(x, y, state, grid, new_positions, new_states, migration_count, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions):
    if state[1] == H and random.random() < hybrid_migration_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        migration_count += 1  # Increment migration count
        new_x, new_y = move_H_cell(x, y, grid)  # Move cell to a new position
        new_states.append((ALIVE, H))
        new_positions.append((new_x, new_y))

        # Update the grid promptly in order to reflect the current grid status for next cells' division and migration in a single update step
        grid[x, y]['primary_state'] = EMPTY
        grid[x, y]['emt_state'] = ''

        grid[new_x, new_y]['primary_state'] = ALIVE
        grid[new_x, new_y]['emt_state'] = H

        if 30 <= new_x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((new_x, new_y))

        return True, migration_count  # Migration occurred
        
    if state[1] == M and random.random() < mesenchymal_migration_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        migration_count += 1  # Increment migration count
        new_x, new_y = move_cells(x, y, new_positions, grid)  # Move cell to a new position
        new_states.append((ALIVE, M))
        new_positions.append((new_x, new_y))

        # Update the grid promptly in order to reflect the current grid status for next cells' division and migration in a single update step
        grid[x, y]['primary_state'] = EMPTY
        grid[x, y]['emt_state'] = ''

        grid[new_x, new_y]['primary_state'] = ALIVE
        grid[new_x, new_y]['emt_state'] = M

        if 30 <= new_x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((new_x, new_y))

        return True, migration_count  # Migration occurred

    return False, migration_count  # Migration didn't happen

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
    
# Define a function for keeping a cell alive
def check_alive(x, y, state, new_positions, new_states, wound_positions):
    new_states.append((ALIVE, state[1]))
    new_positions.append((x, y))  # Keep the original position
    if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
        wound_positions.add((x, y))
    return True  # Cell stays alive

# Function to choose a random action for each cell
def alive_random_action(x, y, state, grid, new_positions, new_states, migration_count, division_probability, death_probability, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions, step, age):
    # If the cell is senescent, it remains in its state and is not processed further
    if ((grid[x, y]["primary_state"], grid[x, y]["emt_state"]) == (SENESCENT, H)) or ((grid[x, y]["primary_state"], grid[x, y]["emt_state"]) == (SENESCENT, E)):
        new_states.append((SENESCENT, state[1]))
        new_positions.append((x, y))
        if 30 <= x <= 69:  # Mark the wound position as updated if applicable
            wound_positions.add((x, y))
        return migration_count
    
    if state[1] == E:
        actions = [
            lambda: (check_division(x, y, state, grid, new_positions, new_states, division_probability, wound_positions), migration_count),
            lambda: (check_death(x, y, state, grid, new_positions, new_states, death_probability, wound_positions), migration_count,),
            lambda: (check_alive(x, y, state, new_positions, new_states, wound_positions), migration_count)
        ]

    if state[1] == H:
        actions = [
            lambda: (check_division(x, y, state, grid, new_positions, new_states, division_probability, wound_positions), migration_count),
            lambda: (check_death(x, y, state, grid, new_positions, new_states, death_probability, wound_positions), migration_count,),
            lambda: check_migration(x, y, state, grid, new_positions, new_states, migration_count, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions),
            lambda: (check_alive(x, y, state, new_positions, new_states, wound_positions), migration_count)
        ]

    if state[1] == M:
        actions = [
            lambda: (check_death(x, y, state, grid, new_positions, new_states, death_probability, wound_positions), migration_count,),
            lambda: check_migration(x, y, state, grid, new_positions, new_states, migration_count, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions),
            lambda: (check_alive(x, y, state, new_positions, new_states, wound_positions), migration_count)
        ]

    # N_ALIVE_DIVIDING = calculate_N_ALIVE_DIVIDING(grid)
    # senescent_prob = ((senescence_rate)*(age+step))/N_ALIVE_DIVIDING
    senescent_prob = senescence_rate*(age+step)
    
    if check_senescence(x, y, state, grid, senescent_prob, new_positions, new_states, wound_positions):
        return migration_count
    else:
        random.shuffle(actions)
        for action in actions:
            success, migration_count = action()
            if success:
                break

    return migration_count  # Return the updated migration_count
