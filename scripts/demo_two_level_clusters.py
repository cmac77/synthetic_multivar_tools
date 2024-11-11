#%% demo_two_level_clusters.py

# Base imports
import os
import sys

# Third-party imports
from pyprojroot import here

# Local imports (ensure 'src' is added to sys.path before importing)
path_root = here()
path_src = os.path.join(path_root, "src")

if path_src not in sys.path:
    sys.path.append(path_src)

from numerical_features import NumericalFeatures, DistanceInfo, ClustersInfo
from figure_generator import FigureGenerator 

#%% Step 1: Setting up paths
# Using prefix-based naming for paths
path_figures = os.path.join(path_root, "results", "figures")

# Ensure the figures directory exists
os.makedirs(path_figures, exist_ok=True)

#%% Step 2: Configuring the synthetic data generation
# Define hierarchical levels
list_levels = [4, 3]  # Two levels: first level with 4 vertices, second level with 3 vertices

# Define distance information
distance_info = DistanceInfo(
    distance=[6, 10],  # Distances for each level
    distance_type=["edge", "edge"]
)

# Define clusters information
clusters_info = ClustersInfo(
    distributions="normal",  # Normal distribution around each centroid
    parameters={"mean": 0, "std_dev": 0.5}  # Default parameters for normal distribution
)

# # Initialize the NumericalFeatures class
# numerical_feature_generator = NumericalFeatures(
#     levels=list_levels,
#     distance_info=distance_info,
#     clusters_info=clusters_info,
#     n_features=3  # Three numerical features for 3D visualization
# )

# Example usage with wildcard
numerical_feature_generator = NumericalFeatures(
    levels=list_levels,
    distance_info=distance_info,
    clusters_info=clusters_info,
    n_features=3,
    selected_centroids=["0", "1", "2", "3*"]  # Selects "0-1" and all of its direct descendants
)

#%% Step 3: Generating synthetic data
# Number of samples to generate
int_n_samples = 1000

# Generate the data
df_numerical_features = numerical_feature_generator.add_numerical_features(
    n_samples=int_n_samples
)
array_data = df_numerical_features.values  # Convert DataFrame to NumPy array

#%% Step 4: Retrieving centroids and distance matrices for visualization
# Extract centroids from the numerical_feature_generator
centroids = numerical_feature_generator.centroids
distance_matrix_levels = numerical_feature_generator.distance_matrix_levels
distance_matrix_global = numerical_feature_generator.distance_matrix_global

#%% Step 5: Visualizing the data using FigureGenerator
# Set the path for saving the first figure (with data points)
path_figure_file = os.path.join(path_figures, "demo_two_level_clusters.png")

# Plot and save the figure with data points
FigureGenerator.plot_data(
    array_data,
    centroids,
    distance_matrix_levels,
    distance_matrix_global,
    title="",
    path_save=path_figure_file
)

# Set the path for saving the second figure (without data points)
path_figure_file_no_data = os.path.join(path_figures, "demo_two_level_clusters_no_data.png")

# Plot and save the figure without data points using the saved axis limits
FigureGenerator.plot_data(
    array_data,
    centroids,
    distance_matrix_levels,
    distance_matrix_global,
    title="",
    path_save=path_figure_file_no_data,
    hide_data=True,
    axis_limits=FigureGenerator.axis_limits  # Use the saved axis limits
)
