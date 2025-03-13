# utils.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap
from constants import *
import os
import pandas as pd
from PIL import Image
import seaborn as sns


# Update color map and legend for the new states
cmap = ListedColormap([
    'white',      # Empty
    'green',      # ALIVE/E
    'palegreen',  # ALIVE/H
    'red',        # Anything/DEAD
    'blue',       # DIVIDING/E
    'purple',     # DIVIDING/H
    'yellow',     # SENESCENT/E
    'orange',     # SENESCENT/H
    'pink'        # Anything/M
])

# Mapping cell states to color indices for the colormap
state_to_index = {
    (EMPTY, ''): 0,
    (ALIVE, 'E'): 1,
    (ALIVE, 'H'): 2,
    (DEAD, ''): 3,
    (DIVIDING, 'E'): 4,
    (DIVIDING, 'H'): 5,
    (SENESCENT, 'E'): 6,
    (SENESCENT, 'H'): 7,
    (ALIVE, 'M'): 8
}

# Create a custom legend based on the defined color map
def create_legend():
    legend_labels = [
        'ALIVE/E', 'ALIVE/H', 'DEAD', 'DIVIDING/E', 'DIVIDING/H', 
        'SENESCENT/E', 'SENESCENT/H', 'M'
    ]
    legend_colors = ['green', 'palegreen', 'red', 'blue', 'purple', 'yellow', 'orange', 'pink']
    return [mpatches.Patch(color=legend_colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]

# Define the structured data type for the grid
dtype = [('primary_state', 'i4'), ('emt_state', 'U6')]  # 'U6' to accommodate the string 'EMPTY'

def update_grid(grid, cell_positions, cell_states, grid_size_x, grid_size_y):
    # Clear the grid by setting it to EMPTY values
    grid['primary_state'] = EMPTY
    grid['emt_state'] = ''
    
    # Update the grid based on cell positions and states
    for pos, state in zip(cell_positions, cell_states):
        grid[pos[0], pos[1]]['primary_state'] = state[0]
        grid[pos[0], pos[1]]['emt_state'] = state[1]

    # Create a color grid for visualization, using the state_to_index mapping
    color_grid = np.array([[state_to_index.get((cell['primary_state'], cell['emt_state']), 0) for cell in row] for row in grid])

    return grid, color_grid

# def visualize_grid(color_grid, step, run_number, senescence_probability, save_images=False):
#     output_dir = 'simulation_images'

#     # Create the plot without axes or any extra elements
#     plt.imshow(color_grid, cmap=cmap, vmin=0, vmax=len(cmap.colors) - 1)
#     plt.axis('off')  # Turn off axis

#     # Save the image if required
#     if save_images:
#         # Ensure the output directory exists
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         # Save the image with a descriptive filename
#         filename = os.path.join(output_dir, f'run_{run_number + 1}_senescence_{senescence_probability:.1e}_step_{step:03d}.png')
#         plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=100)  # Set dpi for 100x100 pixels
#     plt.clf()  # Clear the plot after saving'

def visualize_grid(color_grid, step, run_number, senescence_probability, save_images=False):
    output_dir='simulation_images'

    # Set up the plot with a larger figure size to provide space for the legend
    plt.figure(figsize=(10, 8))  # Increase figure size for more space

        # Plot the grid with the custom colormap
    plt.imshow(color_grid, cmap=cmap, vmin=0, vmax=len(cmap.colors) - 1)
    plt.title(f'Step {step}')
    # plt.legend(handles=create_legend(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save the image if required
    if save_images:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the image with a descriptive filename
        filename = os.path.join(output_dir, f'run_{run_number + 1}_senescence_{senescence_probability:.1e}_step_{step:03d}.png')
        plt.savefig(filename)

    # Display the plot and clear it afterward
    # plt.pause(0.1)
    plt.clf()
    plt.close()

def calculate_permeability(grid):
    # Calculate the permeability
    edge_neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    avg_permeability_sum = 0
    cell_count = 0
    avg_permeability = 0

    # Cache grid dimensions to avoid repeated calls to len(grid)
    grid_rows = len(grid)
    grid_cols = len(grid[0])

    # Loop through every cell in the grid
    for width in range(grid_rows):
        for length in range(grid_cols):
            # Check if the grid cell is not EMPTY or DEAD
            if (grid[width, length]["primary_state"], grid[width, length]["emt_state"]) not in {(EMPTY, ''), (DEAD, '')}:
                cell_count += 1
                single_cell_permeability_sum = 0

                # Check neighboring cells
                for dx, dy in edge_neighbors:
                    nx, ny = width + dx, length + dy
                    # Ensure we stay within grid bounds
                    if 0 <= nx < grid_rows and 0 <= ny < grid_cols:
                        # Check if neighbor is EMPTY or SENESCENT
                        if (grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) in {(EMPTY, ''), (SENESCENT, E), (SENESCENT, H)}:
                            single_cell_permeability_sum += 1

                # Add the permeability score of the current cell
                avg_permeability_sum += single_cell_permeability_sum / 4  # Dividing by 4 to average the permeability score for the cell

    # Only calculate the final average if we have valid cells
    if cell_count > 0:
        avg_permeability = avg_permeability_sum / cell_count
    
    return avg_permeability

# Modified function to create plots for each run
def plot_combined_results(input_dir, output_dir='plot_results_each_run'):
    # Dictionary to store data for each run
    run_data = {}

    # Iterate through all Excel files in the input directory and extract data
    for file in os.listdir(input_dir):
        if file.endswith('.xlsx') and 'division_migration_senescence' in file:
            # Load the results from the Excel file
            filepath = os.path.join(input_dir, file)
            df = pd.read_excel(filepath, engine="openpyxl")
            
            # Verify the dataframe is not empty and contains necessary columns
            if df.empty or 'Age(h)' not in df.columns:
                print(f"Skipping file {file} due to missing data or incorrect format.")
                continue

            # Extract run number from the filename
            run_number = file.split('_run_')[-1].replace('.xlsx', '')
            if run_number not in run_data:
                run_data[run_number] = {
                    'age_list': [],
                    'step_list': [],
                    'division_counts_list': [],
                    'migration_counts_list': [],
                    'avg_permeability_list': [],
                    'wound_area_list': []
                }

            age = df['Age(h)'].iloc[0]
            print(f"Processing file: {file} with senescence probability: {age} for run: {run_number}")

            # Store data for plotting
            run_data[run_number]['age_list'].append(age)
            run_data[run_number]['step_list'].append(df['Step(h)'])
            run_data[run_number]['division_counts_list'].append(df['Division Count'])
            run_data[run_number]['migration_counts_list'].append(df['Migration Count'])
            run_data[run_number]['avg_permeability_list'].append(df['Average Permeability'])
            run_data[run_number]['wound_area_list'].append(df['Wound Area'])

    # Create plots for each run
    for run_number, data in run_data.items():
        age_list = data['age_list']
        step_list = data['step_list']
        division_counts_list = data['division_counts_list']
        migration_counts_list = data['migration_counts_list']
        avg_permeability_list = data['avg_permeability_list']
        wound_area_list = data['wound_area_list']

        # Sort data by senescence probability
        sorted_indices = sorted(range(len(age_list)), key=lambda i: age_list[i])
        age_list = [age_list[i] for i in sorted_indices]
        step_list = [step_list[i] for i in sorted_indices]
        division_counts_list = [division_counts_list[i] for i in sorted_indices]
        migration_counts_list = [migration_counts_list[i] for i in sorted_indices]
        avg_permeability_list = [avg_permeability_list[i] for i in sorted_indices]
        wound_area_list = [wound_area_list[i] for i in sorted_indices]

        # Create a figure with 4 subplots in a 2x2 layout
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Division, Migration, Permeability, and Wound Area vs Step', fontsize=16)

        # Plot Division Count for different senescence probabilities
        for i in range(len(age_list)):
            axs[0, 0].plot(step_list[i], division_counts_list[i], label=f'Age: {age_list[i]:.1e}')
        axs[0, 0].set_xlabel('Step')
        axs[0, 0].set_ylabel('Division Count')
        axs[0, 0].set_title('Division Count vs Step')

        # Plot Migration Count for different senescence probabilities
        for i in range(len(age_list)):
            axs[0, 1].plot(step_list[i], migration_counts_list[i], label=f'Age: {age_list[i]:.1e}')
        axs[0, 1].set_xlabel('Step')
        axs[0, 1].set_ylabel('Migration Count')
        axs[0, 1].set_title('Migration Count vs Step')

        # Plot Average Permeability for different age_list
        for i in range(len(age_list)):
            axs[1, 0].plot(step_list[i], avg_permeability_list[i], label=f'Age: {age_list[i]:.1e}')
        axs[1, 0].set_xlabel('Step')
        axs[1, 0].set_ylabel('Average Permeability')
        axs[1, 0].set_title('Average Permeability vs Step')

        # Plot Wound Area (Empty/Dead Cells Count in Wound) for different senescence probabilities
        for i in range(len(age_list)):
            axs[1, 1].plot(step_list[i], wound_area_list[i], label=f'Age: {age_list[i]:.1e}')
        axs[1, 1].set_xlabel('Step')
        axs[1, 1].set_ylabel('Wound Area (Empty/Dead Cells)')
        axs[1, 1].set_title('Wound Area vs Step')

        # Add legends to all subplots in a sorted order
        for ax in axs.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # Check if there are any handles to add to the legend
                sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: float(x[1].split(": ")[1]))
                sorted_handles, sorted_labels = zip(*sorted_handles_labels)
                ax.legend(sorted_handles, sorted_labels, loc='upper right')

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the combined plot to a file in the new directory
        plot_filename = os.path.join(output_dir, f'combined_plot_run_{run_number}.png')
        plt.savefig(plot_filename)

        # Close the plot
        plt.close()

# # Function to combine and plot results from multiple Excel files
# def plot_combined_results(input_dir, output_dir='plot_combined_results'):
#     # Dictionary to store aggregated data across runs
#     combined_data = {
#         'Age(h)': [],
#         'Step(h)': [],
#         'Division Count': [],
#         'Migration Count': [],
#         'Average Permeability': [],
#         'Wound Area': [],
#         'Senescent Number': [],
#         'Wound Closure Step': []
#     }

#     # Iterate through all Excel files in the input directory
#     for file in os.listdir(input_dir):
#         if file.endswith('.xlsx') and 'division_migration_senescence' in file:
#             # Load the results from the Excel file
#             filepath = os.path.join(input_dir, file)

#             try:
#                 df = pd.read_excel(filepath)

#                 # Ensure required columns exist
#                 required_columns = ['Age(h)', 'Step(h)', 'Division Count', 'Migration Count', 
#                                     'Average Permeability', 'Wound Area', 'Senescent Number', 'Wound Closure Step']
#                 missing_columns = [col for col in required_columns if col not in df.columns]

#                 if df.empty or missing_columns:
#                     print(f"Skipping file {file} due to missing data or incorrect format. Missing columns: {missing_columns}")
#                     continue

#                 # Store data for aggregation
#                 for col in combined_data.keys():
#                     combined_data[col].extend(df[col])

#                 print(f"Processed file: {file}")

#             except Exception as e:
#                 print(f"Error reading file {file}: {e}")
#                 continue

#     # Convert aggregated data to a DataFrame
#     combined_df = pd.DataFrame(combined_data)

#     # Ensure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Save combined data to an Excel file
#     combined_filepath = os.path.join(output_dir, 'combined_results.xlsx')
#     combined_df.to_excel(combined_filepath, index=False)
#     print(f"Combined results saved to {combined_filepath}")

#     # Set Seaborn style for better visuals
#     sns.set(style="whitegrid")

#     # Unique ages for color differentiation
#     unique_ages = sorted(combined_df['Age(h)'].unique())

#     # Create a figure with 4 subplots in a 2x2 layout
#     fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle('Combined Results for Division, Migration, Permeability, and Wound Area', fontsize=16)

#     # Function to plot each Age(h) with a different color
#     def plot_line_graph(ax, y_column, title, ylabel):
#         for age in unique_ages:
#             subset = combined_df[combined_df['Age(h)'] == age]
#             ax.plot(subset['Step(h)'], subset[y_column], label=f'Age: {age}', marker='o', linestyle='-')
#         ax.set_xlabel('Step (h)')
#         ax.set_ylabel(ylabel)
#         ax.set_title(title)
#         ax.legend()

#     # Plot Division Count vs Step
#     plot_line_graph(axs[0, 0], 'Division Count', 'Division Count vs Step', 'Division Count')

#     # Plot Migration Count vs Step
#     plot_line_graph(axs[0, 1], 'Migration Count', 'Migration Count vs Step', 'Migration Count')

#     # Plot Average Permeability vs Step
#     plot_line_graph(axs[1, 0], 'Average Permeability', 'Average Permeability vs Step', 'Average Permeability')

#     # Plot Wound Area vs Step
#     plot_line_graph(axs[1, 1], 'Wound Area', 'Wound Area vs Step', 'Wound Area')

#     # Adjust layout and save plot
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plot_filename = os.path.join(output_dir, 'combined_results_plot.png')
#     plt.savefig(plot_filename)
#     plt.close()

#     print(f"Combined plot saved to {plot_filename}")

def combine_sim_fig(image_dir):
    # List all PNG image files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    # Ensure filenames follow the expected format
    valid_files = [f for f in image_files if len(f.split("_")) > 5]  # Ensure enough parts

    # Construct full file paths
    image_paths = {filename: os.path.join(image_dir, filename) for filename in valid_files}

    # Load images
    images = {filename: Image.open(img) for filename, img in image_paths.items()}

    # Extract unique senescence levels and step numbers
    senescence_levels = sorted(
        {float(f.split("_")[3]) for f in valid_files}, key=float  # Correct index
    )
    steps = sorted(
        {int(f.split("_")[5].replace(".png", "")) for f in valid_files}, key=int  # Correct index
    )

    rows = len(senescence_levels)
    cols = len(steps)

    # Ensure all images have the same size
    widths, heights = zip(*(img.size for img in images.values()))
    max_width = max(widths)
    max_height = max(heights)

    # Resize images to a uniform size
    resized_images = {filename: img.resize((max_width, max_height)) for filename, img in images.items()}

    # Create a blank canvas for the final combined image
    combined_width = max_width * cols
    combined_height = max_height * rows
    final_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))

    # Arrange images in a grid layout
    for row_idx, senescence_prob in enumerate(senescence_levels):
        for col_idx, step in enumerate(steps):
            # Find the corresponding image
            matching_file = next(
                (filename for filename in valid_files
                 if float(filename.split("_")[3]) == senescence_prob and 
                    int(filename.split("_")[5].replace(".png", "")) == step), 
                None
            )
            
            if matching_file:
                img = resized_images[matching_file]
                final_image.paste(img, (col_idx * max_width, row_idx * max_height))

    # Save the final combined image
    output_path = os.path.join(image_dir, "combined_senescent_plot.png")
    final_image.save(output_path)

    # Display the result
    plt.figure(figsize=(12, 10))
    plt.imshow(final_image)
    plt.axis('off')
    plt.show()

    print(f"Combined image saved at: {output_path}")
