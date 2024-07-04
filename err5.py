import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from matplotlib.animation import FuncAnimation

# Function to convert DMS to decimal degrees
def dms_to_dd(degrees, minutes, seconds, direction):
    dd = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

# Function to check if a point is within the given distance from another point using the Haversine formula
def covers(point, target, distance):
    return haversine(point, target) <= distance

# Function to calculate the Haversine distance between two points in kilometers
def haversine(coord1, coord2):
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth radius in kilometers
    r = 6371.0
    return r * c

def is_within_elevation_range(center, terrain_data, bounds):
    lat_index = int((center[0] - bounds[0]) / terrain_data.res[0])
    lon_index = int((center[1] - bounds[2]) / terrain_data.res[1])
    
    # Debugging output
    # print(f"Computed indices: lat_index={lat_index}, lon_index={lon_index}")
    # print(f"Terrain data shape: {terrain_data.shape}")

    # Check if indices are within bounds
    if lat_index < 0 or lat_index >= terrain_data.shape[0] or lon_index < 0 or lon_index >= terrain_data.shape[1]:
        return False
    
    elevation = terrain_data.read(1)[lat_index, lon_index]
    return 0 <= elevation <= 2000

# Fitness function
def fitness(particle, data, distance, terrain_data, bounds):
    total_coverage = 0
    for point in data:
        for center in particle:
            if covers(center, point, distance):
                total_coverage += 1
                break
    return total_coverage

# PSO function
def pso(data, bounds, distance, num_particles, num_iterations, num_centers, terrain_data):
    # Initialize particles
    particles = np.random.uniform(low=[bounds[0], bounds[2]], high=[bounds[1], bounds[3]], size=(num_particles, num_centers, 2))
    velocities = np.random.uniform(low=-1, high=1, size=(num_particles, num_centers, 2))

    # Ensure initial particles are within the elevation range
    for i in range(num_particles):
        for j in range(num_centers):
            while not is_within_elevation_range(particles[i, j], terrain_data, bounds):
                particles[i, j] = np.random.uniform(low=[bounds[0], bounds[2]], high=[bounds[1], bounds[3]], size=2)

    # Initialize personal bests and global best
    p_best = particles.copy()
    p_best_fitness = np.array([fitness(p, data, distance, terrain_data, bounds) for p in particles])
    g_best = p_best[np.argmax(p_best_fitness)]
    g_best_fitness = np.max(p_best_fitness)

    # Initialize particle history
    particle_history = []

    # PSO iterations
    for iteration in range(num_iterations):
        for i in range(num_particles):
            print(iteration)
            # Update velocity
            velocities[i] = (w * velocities[i]
                             + c1 * np.random.rand() * (p_best[i] - particles[i])
                             + c2 * np.random.rand() * (g_best - particles[i]))
            # Update particle position
            particles[i] += velocities[i]
            # Boundary conditions
            particles[i] = np.clip(particles[i], [bounds[0], bounds[2]], [bounds[1], bounds[3]])

            # Ensure particles are within the elevation range
            for j in range(num_centers):
                if not is_within_elevation_range(particles[i, j], terrain_data, bounds):
                    particles[i, j] = np.random.uniform(low=[bounds[0], bounds[2]], high=[bounds[1], bounds[3]], size=2)

            # Update personal best
            current_fitness = fitness(particles[i], data, distance, terrain_data, bounds)
            if current_fitness > p_best_fitness[i]:
                p_best[i] = particles[i]
                p_best_fitness[i] = current_fitness

                # Update global best
                if current_fitness > g_best_fitness:
                    g_best = particles[i]
                    g_best_fitness = current_fitness

        # Store particle positions for animation
        particle_history.append(particles.copy())

    # Return the best particles, best fitness, and particle history
    return g_best, g_best_fitness, particle_history

# # Function to check if a point is within the given distance from another point using Haversine formula
# def covers(point, target, distance):
#     return haversine(point, target) <= distance

# # Function to evaluate fitness of a particle
# def fitness(particle, data, distance, terrain_data, bounds):
#     total_coverage = 0
#     for point in data:
#         for center in particle:
#             if covers(center, point, distance):
#                 lat_index = int((point[0] - bounds[0]) / terrain_data.res[0])
#                 lon_index = int((point[1] - bounds[2]) / terrain_data.res[1])
#                 elevation = terrain_data.read(1)[lat_index, lon_index]
#                 if 0 <= elevation <= 2000:
#                     total_coverage += 1
#                     break
#     return total_coverage


# Define the PSO parameters
num_particles = 30  # Number of particles in the swarm
num_iterations = 100  # Number of iterations to run the PSO
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive (particle) weight
c2 = 1.5  # Social (swarm) weight

# Define the bounds for the search space in decimal degrees
bounds = [
    min(dms_to_dd(28, 43, 10, 'N'), dms_to_dd(29, 57, 52, 'N'), dms_to_dd(31, 24, 35, 'N'), dms_to_dd(30, 17, 49, 'N')),
    max(dms_to_dd(28, 43, 10, 'N'), dms_to_dd(29, 57, 52, 'N'), dms_to_dd(31, 24, 35, 'N'), dms_to_dd(30, 17, 49, 'N')),  # Latitude bounds
    min(dms_to_dd(79, 58, 10, 'E'), dms_to_dd(77, 31, 10, 'E'), dms_to_dd(78, 32, 28, 'E'), dms_to_dd(80, 58, 9, 'E')),
    max(dms_to_dd(79, 58, 10, 'E'), dms_to_dd(77, 31, 10, 'E'), dms_to_dd(78, 32, 28, 'E'), dms_to_dd(80, 58, 9, 'E'))   # Longitude bounds
]

# Define the data points (converted to decimal degrees)
data = np.array([
    [dms_to_dd(30, 18, 40, 'N'), dms_to_dd(78, 2, 9, 'E')],
    [dms_to_dd(29, 56, 43, 'N'), dms_to_dd(78, 9, 4, 'E')],
    [dms_to_dd(29, 52, 43, 'N'), dms_to_dd(77, 53, 15, 'E')],
    [dms_to_dd(30, 13, 32, 'N'), dms_to_dd(78, 46, 58, 'E')],
    [dms_to_dd(30, 8, 42, 'N'), dms_to_dd(78, 45, 59, 'E')],
    [dms_to_dd(30, 24, 4, 'N'), dms_to_dd(78, 27, 51, 'E')],
    [dms_to_dd(30, 33, 44, 'N'), dms_to_dd(79, 33, 46, 'E')],
    [dms_to_dd(29, 34, 41, 'N'), dms_to_dd(79, 39, 23, 'E')],
    [dms_to_dd(29, 34, 24, 'N'), dms_to_dd(80, 15, 18, 'E')],
    [dms_to_dd(30, 46, 50, 'N'), dms_to_dd(79, 16, 23, 'E')],
    [dms_to_dd(31, 0, 38, 'N'), dms_to_dd(79, 3, 27, 'E')],
    [dms_to_dd(31, 6, 33, 'N'), dms_to_dd(78, 51, 35, 'E')],
    [dms_to_dd(31, 5, 9, 'N'), dms_to_dd(78, 35, 36, 'E')],
    [dms_to_dd(31, 1, 12, 'N'), dms_to_dd(78, 33, 7, 'E')]
])

# Set the distance for coverage in kilometers
distance = 50  # Assuming 50 km for the coverage radius

# Load the terrain data
terrain_file = "/Users/parasdhiman/Desktop/DRDO/output_SRTMGL1 2.tif"  # Replace with your actual file path
terrain_data = rasterio.open(terrain_file)

# Number of centers to find
num_centers = 3

# Find the best centers using PSO
best_points, _, particle_history = pso(data, bounds, distance, num_particles, num_iterations, num_centers, terrain_data)

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
    ax.set_title(f'PSO Optimization for Center Locations (Iteration {frame + 1}/{num_iterations})')

    cbar.set_label('Elevation (m)')

    ax.legend()

# Animate the PSO process
ani = FuncAnimation(fig, update, frames=len(particle_history), interval=200, repeat=False)
plt.show()
