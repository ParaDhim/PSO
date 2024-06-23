import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# Function to convert DMS to decimal degrees
def dms_to_dd(degrees, minutes, seconds, direction):
    dd = degrees + minutes/60 + seconds/3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

# Function to calculate the Haversine distance between two points in kilometers
def haversine(coord1, coord2):
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth radius in kilometers
    r = 6371.0
    return r * c

# Function to check if a point is within the given distance from another point using Haversine formula
def covers(point, target, distance):
    return haversine(point, target) <= distance

# PSO Parameters
num_particles = 30  # Number of particles in the swarm
num_iterations = 5000  # Number of iterations to run the PSO
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive (particle) weight
c2 = 1.5  # Social (swarm) weight

# Function to evaluate fitness of a particle
def fitness(particle, data, distance):
    return np.sum([covers(particle, point, distance) for point in data])

# PSO Algorithm to find the best center
def pso(data, bounds, distance, num_particles, num_iterations):
    # Initialize particles
    particles = np.random.uniform(low=[bounds[0], bounds[2]], high=[bounds[1], bounds[3]], size=(num_particles, 2))
    velocities = np.random.uniform(low=-1, high=1, size=(num_particles, 2))

    # Initialize personal bests and global best
    p_best = particles.copy()
    p_best_fitness = np.array([fitness(p, data, distance) for p in particles])
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
            current_fitness = fitness(particles[i], data, distance)
            if current_fitness > p_best_fitness[i]:
                p_best[i] = particles[i]
                p_best_fitness[i] = current_fitness

                # Update global best
                if current_fitness > g_best_fitness:
                    g_best = particles[i]
                    g_best_fitness = current_fitness

    return g_best, g_best_fitness

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

# Function to check overall coverage of a set of centers
def overall_coverage(centers, data, distance):
    covered_points = []
    for center in centers:
        covered_points.extend([point for point in data if covers(center, point, distance)])
    covered_points = np.unique(covered_points, axis=0)
    return len(covered_points)

# Load the terrain data
terrain_file = "/Users/parasdhiman/Desktop/DRDO/output_SRTMGL1 2.tif"
terrain_data = rasterio.open(terrain_file)

# Iterate over different numbers of particles/centers
for num_centers in range(3, 4):
    # Initialize the uncovered points
    uncovered = data.copy()
    centers = []

    # Find the best centers using PSO
    for _ in range(num_centers):
        best_point, _ = pso(uncovered, bounds, distance, num_particles, num_iterations)
        centers.append(best_point)
        covered_points = [target for target in uncovered if covers(best_point, target, distance)]
        uncovered = np.array([point for point in uncovered if not any(np.array_equal(point, covered) for covered in covered_points)])

    # Calculate overall coverage
    coverage = overall_coverage(centers, data, distance)

    # Output the results
    print(f"Number of centers: {num_centers}, Points covered: {coverage}")

    # Plotting the points, centers, and circles
    plt.figure(figsize=(10, 8))

    # Show the terrain data with custom elevation-based colormap
    terrain_image = show((terrain_data, 1), ax=plt.gca(), cmap='terrain')

    # Plot points, centers, and circles
    plt.scatter(data[:, 1], data[:, 0], c='white', label='Points')
    for center in centers:
        plt.scatter(center[1], center[0], c='red', marker='x', s=100, label='Center')
        circle = plt.Circle((center[1], center[0]), distance / 111, color='r', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    # Add color bar for terrain data
    cbar = plt.colorbar(terrain_image.get_children()[0], ax=plt.gca(), fraction=0.03)
    cbar.set_label('Elevation (meters)')

    plt.legend()
    plt.title(f'PSO with {num_centers} Centers: Points and Coverage with {distance} km distance')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

