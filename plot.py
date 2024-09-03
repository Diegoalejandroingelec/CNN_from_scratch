import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 6))  # Widen the figure for better spacing

# Define the layers and their positions
layers = [
    {"name": "Input\n(1, 28, 28)", "pos": (0, 0)},
    {"name": "Conv2D\n(6, 26, 26)", "pos": (4.5, 0)},  # Adjusted spacing
    {"name": "Sigmoid\n(6, 26, 26)", "pos": (9, 0)},
    {"name": "Flatten\n(4056, 1)", "pos": (13.5, 0)},
    {"name": "FC\n(256)", "pos": (18, 0)},
    {"name": "Sigmoid\n(256)", "pos": (22.5, 0)},
    {"name": "FC\n(2)", "pos": (27, 0)},
    {"name": "Sigmoid\n(2)", "pos": (31.5, 0)},
]

# Draw the layers as rectangles (larger boxes)
for layer in layers:
    ax.add_patch(Rectangle(layer["pos"], 2.5, 1.5, edgecolor='black', facecolor='lightblue'))  # Increased box size
    ax.text(layer["pos"][0] + 1.25, layer["pos"][1] + 0.75, layer["name"], 
            horizontalalignment='center', verticalalignment='center', fontsize=10)

# Draw arrows between the layers (thinner and adjusted to end at the box edge)
for i in range(len(layers) - 1):
    ax.add_patch(FancyArrow(layers[i]["pos"][0] + 2.5, layers[i]["pos"][1] + 0.75, 
                            1.72, 0, width=0.05, head_width=0.15, head_length=0.3, color='black'))

# Set limits and hide axes
ax.set_xlim(-1, 37)
ax.set_ylim(-2, 3)
ax.axis('off')

# Show the plot
plt.show()