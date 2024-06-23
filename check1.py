import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Open the raster file
with rasterio.open('/Users/parasdhiman/Downloads/output_SRTMGL1 2.tif') as src:
    terrain = src.read(1)  # Read the terrain data
    transform = src.transform  # Get the transformation matrix

# Find radar position based on highest elevation
radar_position = np.unravel_index(np.argmax(terrain), terrain.shape)

# Calculate line of sight visibility
def line_of_sight(terrain, radar_position):
    visible = np.zeros_like(terrain, dtype=bool)
    rows, cols = terrain.shape
    radar_x, radar_y = radar_position
    
    for row in range(rows):
        for col in range(cols):
            if terrain[row, col] >= terrain[radar_x, radar_y]:
                # If the terrain point is at or above the radar height, it's visible
                visible[row, col] = True
                
    return visible

visibility = line_of_sight(terrain, radar_position)

# Determine suitable radar locations based on visibility
suitable_locations = np.where(visibility)

# Output radar position
print("Radar position:", radar_position)

# Output suitable radar locations
for loc in zip(*suitable_locations):
    print("Radar can be placed at:", loc)
    
# Plot the terrain data
plt.figure(figsize=(10, 8))
plt.imshow(terrain, cmap='terrain', extent=[transform[2], transform[2] + transform[0] * terrain.shape[1], 
                                             transform[5] + transform[4] * terrain.shape[0], transform[5]])
plt.colorbar(label='Elevation (m)')
plt.title('Terrain with Radar Position')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Plot radar position
plt.scatter(transform[2] + radar_position[1] * transform[0], transform[5] + radar_position[0] * transform[4], 
            color='red', marker='o', label='Radar Position')

plt.legend()
plt.show()