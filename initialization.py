# initialization.py

import numpy as np
from constants import EMPTY, ALIVE
from constants import E

# Define a structured data type for grid cells
dtype = [('primary_state', 'i4'), ('emt_state', 'U1')]  # 'i4' for integer, 'U1' for string

def initialize_grid(grid_size_x, grid_size_y):
    # Create an empty structured grid with specified dtype
    grid = np.empty((grid_size_x, grid_size_y), dtype=dtype)
    
    # Fill the grid with EMPTY values (primary_state=EMPTY, emt_state='')
    grid['primary_state'] = EMPTY
    grid['emt_state'] = ''  # Empty string to indicate no EMT state
    
    return grid

def initialize_cells(grid_size_x, grid_size_y):
    # Left block of cells (first 30 grid points)
    left_block = [(j, i) for i in range(30) for j in range(grid_size_y)]
    # Right block of cells (last 30 grid points)
    right_block = [(j, i) for i in range(70, grid_size_x) for j in range(grid_size_y)]

    # Combine both blocks
    initial_positions = np.array(left_block + right_block)

    # Initialize all cells in the initial positions with (ALIVE, E)
    initial_states = np.array([(ALIVE, E)] * len(initial_positions), dtype=dtype)
    print(initial_states)

    return initial_positions, initial_states
