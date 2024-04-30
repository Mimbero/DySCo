import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def custom_red_blue_colormap():
    # Define the colors
    colors = [
        (1.0, 0.0, 0.0),   # Red
        (0.0, 0.0, 1.0)    # Blue
    ]

    # Create a colormap from the defined colors
    return mcolors.LinearSegmentedColormap.from_list('custom_red_blue', colors, N=256)

def blue_to_red_colormap():
    cmap = plt.cm.RdBu
    colors = cmap(np.linspace(0, 1, 256))
    return mcolors.LinearSegmentedColormap.from_list('custom_blue_to_red', colors)

def adjust_saturation_brightness(cmap):
    # Adjust saturation and brightness for red
    red_multiplier = 1
    red_addition = 0.1
    cmap_colors = cmap(np.linspace(0, 1, 256))
    cmap_colors[:128, :3] = np.clip(cmap_colors[:128, :3] * red_multiplier + red_addition, 0, 1)

    # Adjust saturation and brightness for blue
    blue_multiplier = 0.7
    blue_addition = 0.3
    cmap_colors[128:, :3] = np.clip(cmap_colors[128:, :3] * blue_multiplier + blue_addition, 0, 1)

    # Create a colormap from the adjusted colors
    return mcolors.LinearSegmentedColormap.from_list('adjusted_custom_colormap', cmap_colors)


def custom_red_blue_colormap_SAT():
    # Define the colors
    colors = [
        (1.0, 0.0, 0.0),   # Red
        (0.0, 0.0, 1.0)    # Blue
    ]

    # Create a colormap from the defined colors
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_red_blue', colors, N=256)

    # Adjust saturation and brightness for red
    red_multiplier = 0.9
    red_addition = 0.4
    cmap_colors = cmap(np.linspace(0, 1, 256))
    cmap_colors[:128, :3] = np.clip(cmap_colors[:128, :3] * red_multiplier + red_addition, 0, 1)

    # Adjust saturation and brightness for blue
    blue_multiplier = 0.9
    blue_addition = 0.4
    cmap_colors[128:, :3] = np.clip(cmap_colors[128:, :3] * blue_multiplier + blue_addition, 0, 1)

    # Create a colormap from the adjusted colors
    return mcolors.LinearSegmentedColormap.from_list('custom_pastel_blue_to_red', cmap_colors)

# Test the custom colormap
# cmap = custom_red_blue_colormap()
# cmap = blue_to_red_colormap()


cmap = custom_red_blue_colormap_SAT()

plt.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, aspect='auto')
plt.colorbar()
plt.show()