# main.py

from simulation import run_simulation
from constants import *
from utils import plot_combined_results, combine_sim_fig

if __name__ == "__main__":

     # Directory where Excel files are saved
    input_dir = '/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/EMT_Alon_Model'  # Current directory

    try:
        for age in ages:
            run_simulation(age, NUM_STEPS)

        # Generate the combined plots for all senescence probabilities
        plot_combined_results(input_dir)
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # image_dir = "/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/EMT_Alon_Model/Image_Creation/"
    # combine_sim_fig(image_dir)
