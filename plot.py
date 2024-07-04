import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.animation import FuncAnimation
import pickle

# Define the file path to load the parameters
param_file_path = 'pso_params_6chc.pkl'

# Load the parametric information
with open(param_file_path, 'rb') as f:
    params = pickle.load(f)

particle_history = params['particle_history']
bounds = params['bounds']
data = params['data']
best_points = params['best_points']
distance = params['distance']

# Load the terrain data
terrain_file = "/Users/parasdhiman/Desktop/DRDO/output_SRTMGL1 2.tif"  # Replace with your actual file path
terrain_data = rasterio.open(terrain_file)

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))

# Show the terrain data with custom elevation-based colormap
terrain_image = ax.imshow(terrain_data.read(1), extent=[bounds[2], bounds[3], bounds[0], bounds[1]], cmap='terrain', origin='upper')

# Add color bar for elevation
cbar = plt.colorbar(terrain_image, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Elevation (m)')

def update(frame):
    ax.clear()
    ax.imshow(terrain_data.read(1), extent=[bounds[2], bounds[3], bounds[0], bounds[1]], cmap='terrain', origin='upper')
    ax.scatter(data[:, 1], data[:, 0], c='white', label='Points')

    particles = particle_history[frame]
    for particle in particles:
        ax.scatter(particle[:, 1], particle[:, 0], c='blue', alpha=0.5)

    if frame == len(particle_history) - 1:
        for center in best_points:
            ax.scatter(center[1], center[0], c='red', marker='x', s=100, label='Center')
            circle = plt.Circle((center[1], center[0]), distance / 111, color='blue', fill=False)
            ax.add_artist(circle)

    ax.set_xlim(bounds[2], bounds[3])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'PSO Optimization for Center Locations (Iteration {frame + 1}/{len(particle_history)})')

    cbar.set_label('Elevation (m)')

    ax.legend()

# Animate the PSO process
ani = FuncAnimation(fig, update, frames=len(particle_history), interval=200)
ani.save('pso_optimization6chc.mp4', writer='ffmpeg', fps=10) 

# Show the animation
# plt.show()

