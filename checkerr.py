import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# Function to convert DMS to decimal degrees
def dms_to_dd(degrees, minutes, seconds, direction):
    dd = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

# Function to calculate the Haversine distance between two points in kilometers
def haversine(coord1, coord2):
    lat1, lon1 = np.radians(coord1[:2])  # Unpack the first two elements of coord1
    lat2, lon2 = np.radians(coord2[:2])  # Unpack the first two elements of coord2

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth radius in kilometers
    r = 6371.0
    return r * c

# Function to check if a point is within the given distance from another point using Haversine formula
def covers(point, target, distance):
    return haversine(point[:2], target[:2]) <= distance

# Function to perform line of sight check between center and point
def has_line_of_sight(terrain_data, center, point):
    # Example implementation for line of sight check
    center_height = next(terrain_data.sample([(center[0], center[1])]))[0]
    point_height = next(terrain_data.sample([(point[0], point[1])]))[0]
    max_height_difference = 2000  # Maximum allowable height difference in meters

    # Check if height difference exceeds the allowable limit
    if abs(center_height - point_height) > max_height_difference:
        return False

    # Hypothetical implementation for line of sight check (could be more complex based on terrain data)
    # Here, assuming direct line of sight is clear if height difference constraint is met
    return True

# Function to evaluate fitness of a particle (center)
def fitness(particle, data, distance, terrain_data, max_height):
    # Sample elevation data for the particle (center)
    center_height = next(terrain_data.sample([(particle[0], particle[1])]))[0]

    valid_points = []

    for point in data:
        # Sample elevation data for the point
        point_height = next(terrain_data.sample([(point[0], point[1])]))[0]

        # Check if point is within distance and height constraints
        if (haversine(particle[:2], point[:2]) <= distance and
                abs(center_height - point_height) <= max_height and
                center_height <= max_height and
                point_height <= max_height and 
                has_line_of_sight(terrain_data, particle, point)):
            valid_points.append(point)

    return len(valid_points)

# PSO Algorithm to find the best centers
def pso(data, bounds, distance, num_particles, num_iterations, terrain_data, max_height):
    # Initialize particles
    particles = np.random.uniform(low=[bounds[0], bounds[2]], high=[bounds[1], bounds[3]], size=(num_particles, 2))
    velocities = np.random.uniform(low=-1, high=1, size=(num_particles, 2))

    # Initialize personal bests and global best
    p_best = particles.copy()
    p_best_fitness = np.array([fitness(p, data, distance, terrain_data, max_height) for p in particles])
    g_best = p_best[np.argmax(p_best_fitness)]
    g_best_fitness = np.max(p_best_fitness)

    # PSO iterations
    for _ in range(num_iterations):
        for i in range(num_particles):
            # Update velocity
            velocities[i] = (w * velocities[i]
                             + c1 * np.random.rand() * (p_best[i] - particles[i])
                             + c2 * np.random.rand() * (g_best - particles[i]))
            # Update particle position
            particles[i] += velocities[i]
            # Boundary conditions
            particles[i] = np.clip(particles[i], [bounds[0], bounds[2]], [bounds[1], bounds[3]])

            # Update personal best
            current_fitness = fitness(particles[i], data, distance, terrain_data, max_height)
            if current_fitness > p_best_fitness[i]:
                p_best[i] = particles[i]
                p_best_fitness[i] = current_fitness

                # Update global best
                if current_fitness > g_best_fitness:
                    g_best = particles[i]
                    g_best_fitness = current_fitness

    return g_best, g_best_fitness

# Function to check overall coverage of a set of centers
def overall_coverage(centers, data, distance, terrain_data):
    covered_points = []

    for center in centers:
        for point in data:
            if covers(center, point, distance) and has_line_of_sight(terrain_data, center, point):
                covered_points.append(tuple(point))

    covered_points = np.unique(covered_points, axis=0)
    return len(covered_points)

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
terrain_file = "/Users/parasdhiman/Desktop/DRDO/output_SRTMGL1 2.tif"
terrain_data = rasterio.open(terrain_file)

# PSO Parameters
num_iterations = 2000  # Number of iterations to run the PSO
num_centers = 3 
# PSO Parameters (Continued)
num_particles = 30  # Number of particles in the swarm
w = 0.5  # Inertia weight
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter

# Maximum allowable height difference in meters
max_height = 2000

# Run PSO to find the best centers
best_centers = []
best_coverage = 0

for _ in range(num_centers):
    center, fitness_value = pso(data, bounds, distance, num_particles, num_iterations, terrain_data, max_height)
    best_centers.append(center)
    best_coverage += overall_coverage([center], data, distance, terrain_data)

# Print the best centers and their coverage
print("Best Centers:")
for center in best_centers:
    print(center)

print("Best Coverage:", best_coverage)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Visualize the terrain data
cmap = 'terrain'  # You can choose any colormap suitable for terrain visualization
ax.imshow(terrain_data, cmap=cmap, origin='lower')
ax.set_title('Terrain Data')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Plot the data points
ax.scatter(data[:, 1], data[:, 0], color='red', label='Data Points')

# Plot the best centers
for center in best_centers:
    ax.scatter(center[1], center[0], color='blue', label='Best Centers', marker='^')

# Add legend
ax.legend()

# Show the plot
plt.show()
