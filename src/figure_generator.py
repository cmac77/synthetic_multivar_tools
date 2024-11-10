
#%%  figure_generator.py

# Base imports
import os

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

class FigureGenerator:
    """
    A class for generating flexible 2D or 3D visualizations of synthetic data clusters.

    Attributes:
        axis_limits (tuple): Stores the axis limits from the last generated plot.

    Methods:
        plot_data: Plots data in either 2D or 3D, marking centroids with red x's and connecting them with red dotted lines.
        plot_global_edges: Connects all vertices across all levels.
        plot_level_edges: Connects vertices within the same level.
        plot_sibling_edges: Connects sibling vertices (default method), with special handling for Level 0.
    """

    # Initialize the axis_limits attribute
    axis_limits = None

    @staticmethod
    def plot_data(data, centroids, distance_matrix_levels, distance_matrix_global, title="Data Visualization", path_save=None, targets=None, hide_data=False, axis_limits=None):
        """
        Plot data in 2D or 3D, connecting centroids with different methods and marking distances.
        Optionally, visualize relationships to target values or labels.

        Args:
            data (numpy.ndarray or pandas.DataFrame): The data to plot.
            centroids (dict): A dictionary of centroids with keys as labels and values as coordinate arrays.
            distance_matrix_levels (dict): Distance matrix information for each level.
            distance_matrix_global (dict): Global distance matrix for all vertices.
            title (str): Title of the plot.
            path_save (str, optional): Path to save the plot. If None, the plot will only be displayed.
            targets (numpy.ndarray or pandas.Series, optional): Target values or labels to visualize relationships.
            hide_data (bool, optional): Whether to hide data points in the plot. Default is False.
            axis_limits (tuple, optional): A tuple containing axis limits (x_min, x_max, y_min, y_max, [z_min, z_max]).
                                           Must match the number of dimensions in the plot.
        
        Raises:
            ValueError: If the number of dimensions is not 2 or 3 or if axis_limits do not match the number of dimensions.
        """
        # Determine the number of dimensions
        num_dimensions = data.shape[1]

        # Check if the number of dimensions is valid for plotting
        if num_dimensions not in [2, 3]:
            raise ValueError("Only 2D or 3D plotting is supported.")

        # Check if axis_limits are provided and valid
        if axis_limits is not None:
            expected_length = 4 if num_dimensions == 2 else 6
            if len(axis_limits) != expected_length:
                raise ValueError(f"axis_limits must have {expected_length} elements for a {num_dimensions}D plot.")

        # Create a new figure
        fig = plt.figure(figsize=(10, 8))

        # Set up color mapping if targets are provided
        color_map = None
        if targets is not None:
            unique_targets = np.unique(targets)
            color_map = cm.get_cmap("viridis", len(unique_targets)) if len(unique_targets) > 2 else cm.get_cmap("coolwarm")

        if num_dimensions == 3:
            ax = fig.add_subplot(111, projection='3d')
            # Plot data points if hide_data is False
            if not hide_data:
                if targets is not None:
                    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=targets, cmap=color_map, alpha=0.6)
                    fig.colorbar(scatter, ax=ax, label="Target")
                else:
                    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', marker='o', alpha=0.6)
            
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")

            # Heuristic optimal viewpoint
            elevation, azimuth = FigureGenerator._calculate_optimal_viewpoint(ax, centroids)
            ax.view_init(elev=elevation, azim=azimuth)

            # Set axis limits if provided
            if axis_limits:
                ax.set_xlim(axis_limits[0], axis_limits[1])
                ax.set_ylim(axis_limits[2], axis_limits[3])
                ax.set_zlim(axis_limits[4], axis_limits[5])
            else:
                # Save the automatically determined axis limits
                FigureGenerator.axis_limits = (
                    ax.get_xlim()[0], ax.get_xlim()[1],
                    ax.get_ylim()[0], ax.get_ylim()[1],
                    ax.get_zlim()[0], ax.get_zlim()[1]
                )
        else:
            ax = fig.add_subplot(111)
            # Plot data points if hide_data is False
            if not hide_data:
                if targets is not None:
                    scatter = ax.scatter(data[:, 0], data[:, 1], c=targets, cmap=color_map, alpha=0.6)
                    fig.colorbar(scatter, ax=ax, label="Target")
                else:
                    ax.scatter(data[:, 0], data[:, 1], c='blue', marker='o', alpha=0.6)
            
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

            # Set axis limits if provided
            if axis_limits:
                ax.set_xlim(axis_limits[0], axis_limits[1])
                ax.set_ylim(axis_limits[2], axis_limits[3])
            else:
                # Save the automatically determined axis limits
                FigureGenerator.axis_limits = (
                    ax.get_xlim()[0], ax.get_xlim()[1],
                    ax.get_ylim()[0], ax.get_ylim()[1]
                )

        # Plot centroids with thicker Xs
        for key, centroid in centroids.items():
            if num_dimensions == 3:
                ax.scatter(centroid[0], centroid[1], centroid[2], c='red', marker='x', s=100, linewidths=1)
            else:
                ax.scatter(centroid[0], centroid[1], c='red', marker='x', s=100, linewidths=1)

        # Plot edges using the default method (siblings)
        FigureGenerator.plot_sibling_edges(ax, centroids, distance_matrix_levels)

        # Set the title and show or save the plot
        plt.title(title)
        if path_save:
            plt.savefig(path_save, dpi=300)
            print(f"Figure saved to {path_save}")
        plt.show()


    @staticmethod
    def plot_global_edges(ax, centroids, distance_matrix_global):
        """Connect all vertices across all levels using thicker red dotted lines."""
        keys = distance_matrix_global['keys']
        distance_matrix = distance_matrix_global['distance_matrix']

        for i, key_i in enumerate(keys):
            for j, key_j in enumerate(keys):
                if i < j:  # Avoid duplicate lines
                    c1 = centroids[key_i]
                    c2 = centroids[key_j]
                    if len(c1) == 3:  # 3D plot
                        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 'r--', linewidth=1)
                    else:  # 2D plot
                        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'r--', linewidth=1)

    @staticmethod
    def plot_level_edges(ax, centroids, distance_matrix_levels):
        """Connect vertices within the same level using thicker red dotted lines."""
        for level, info in distance_matrix_levels.items():
            keys = info['keys']
            distance_matrix = info['distance_matrix']

            for i, key_i in enumerate(keys):
                for j, key_j in enumerate(keys):
                    if i < j:  # Avoid duplicate lines
                        c1 = centroids[key_i]
                        c2 = centroids[key_j]
                        if len(c1) == 3:  # 3D plot
                            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 'r--', linewidth=1)
                        else:  # 2D plot
                            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'r--', linewidth=1)

    @staticmethod
    def plot_sibling_edges(ax, centroids, distance_matrix_levels):
        """Connect sibling vertices within the same level using thicker red dotted lines, with special handling for Level 0."""
        for level, info in distance_matrix_levels.items():
            keys = info['keys']
            distance_matrix = info['distance_matrix']

            # Special handling for Level 0: Connect all vertices as siblings and label distances
            if level == 0:
                for i, key_i in enumerate(keys):
                    for j, key_j in enumerate(keys):
                        if i < j:  # Avoid duplicate lines
                            c1 = centroids[key_i]
                            c2 = centroids[key_j]
                            if len(c1) == 3:  # 3D plot
                                ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 'r--', linewidth=1)
                            else:  # 2D plot
                                ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'r--', linewidth=1)

                            # Calculate and annotate the distance with bold red font
                            distance = np.linalg.norm(np.array(c1) - np.array(c2))
                            mid_point = (np.array(c1) + np.array(c2)) / 2
                            if len(c1) == 3:  # 3D plot
                                ax.text(mid_point[0], mid_point[1], mid_point[2], f"{distance:.2f}", color='red', fontsize=8, fontweight='bold', zorder=10)
                            else:  # 2D plot
                                ax.text(mid_point[0], mid_point[1], f"{distance:.2f}", color='red', fontsize=8, fontweight='bold', zorder=10)
            else:
                # Group keys by parent based on the hyphen count for other levels
                sibling_groups = {}
                for key in keys:
                    parent = '-'.join(key.split('-')[:-1]) if '-' in key else key
                    sibling_groups.setdefault(parent, []).append(key)

                # Connect siblings within each group and label distances
                for siblings in sibling_groups.values():
                    for i, key_i in enumerate(siblings):
                        for j, key_j in enumerate(siblings):
                            if i < j:  # Avoid duplicate lines
                                c1 = centroids[key_i]
                                c2 = centroids[key_j]
                                if len(c1) == 3:  # 3D plot
                                    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 'r--', linewidth=1)
                                else:  # 2D plot
                                    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'r--', linewidth=1)

                                # Calculate and annotate the distance with bold red font
                                distance = np.linalg.norm(np.array(c1) - np.array(c2))
                                mid_point = (np.array(c1) + np.array(c2)) / 2
                                if len(c1) == 3:  # 3D plot
                                    ax.text(mid_point[0], mid_point[1], mid_point[2], f"{distance:.2f}", color='red', fontsize=8, fontweight='bold', zorder=10)
                                else:  # 2D plot
                                    ax.text(mid_point[0], mid_point[1], f"{distance:.2f}", color='red', fontsize=8, fontweight='bold', zorder=10)

    @staticmethod
    def _calculate_optimal_viewpoint(ax, centroids):
        """Calculate an optimal viewpoint for 3D visualization."""
        centroid_array = np.array(list(centroids.values()))
        center_of_mass = np.mean(centroid_array, axis=0)
        centroid_array_homogeneous = np.hstack([centroid_array, np.ones((centroid_array.shape[0], 1))])

        elevation = 30
        azimuth = 45
        max_spread = 0

        for elev in range(20, 80, 10):
            for azim in range(0, 360, 45):
                ax.view_init(elev=elev, azim=azim)
                projected_points = ax.get_proj()
                spread = np.linalg.norm((centroid_array_homogeneous @ projected_points.T)[:, :3] - center_of_mass)

                if spread > max_spread:
                    max_spread = spread
                    elevation, azimuth = elev, azim

        return elevation, azimuth