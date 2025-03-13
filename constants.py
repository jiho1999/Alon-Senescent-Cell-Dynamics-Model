# Cell states
ALIVE = 1
DEAD = 0
DIVIDING = 2
SENESCENT = 3
EMPTY = -1  # New constant for empty spots

# EMT states
E = 'E'  # Epithelial
H = 'H'  # Hybrid
M = 'M'  # Mesenchymal

# Parameters for probabilities
division_probability = 0.03
hybrid_migration_probability = 0.9
mesenchymal_migration_probability = 0.99
senescent_migration_probability = 0.45
death_probability = 0.0003
# constant_senescence_probability =[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
senescence_rate = 0.00000069
removal_rate_ß = 0.0125
# age of 0, 0.5, 1, 1.5, 2, and 2.5 years old mice
ages = [0, 4380, 8760, 13140, 17520, 21900]
# ages = [21900]
# Probability of mesenchymal cells turn into epithelial cell
M_to_E_probability = 0.1

# Size of the grid
grid_size_x = 100
grid_size_y = 100

# Number of steps
NUM_STEPS = 500

"""
1. Senescent cell accumulation
 - Senescence probability: ητ/ # of ALIVE & DIVIDING cells = (((6.94*10^-7)/hour^2)(t_age + t_step)) / # of ALIVE & DIVIDING cells
 - Senescence probability when dividing: senescence probability * alpha
   (The value of alpha is unknown)

2. Senescent cell removal
 - Removal probability = B*(1/(k+x)) / # of SENESCENT cell = (0.0125*hour^-1)*(1/1+ % of senescent cell) / # of SENESCENT cells
 B = 0.0125/hour (0.3/day)
 k = 1% of all living cell,
 x = % of senescent cell among living cell)

 3. Senesent cell dynamics
 dX/dt = production - removal = (6.94*10^-7)t - 0.0125(1/k+x)
 * time is t1(step) + t2(age)
"""
