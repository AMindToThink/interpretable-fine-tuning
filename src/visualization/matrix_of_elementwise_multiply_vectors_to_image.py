import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb

def visualize_matrix_hsv(matrix):
    """
    Visualize a matrix where:
    - 0 values are black
    - Positive values are blue with intensity based on magnitude
    - Negative values are red with intensity based on magnitude
    
    Args:
        matrix (numpy.ndarray): The input matrix to visualize
    """
    # Get matrix dimensions
    if len(matrix.shape) == 1:
        # Convert vector to matrix with one column
        matrix = matrix.reshape(-1, 1)
    
    rows, cols = matrix.shape
    
    # Create an HSV image (Hue, Saturation, Value)
    hsv_image = np.zeros((rows, cols, 3))
    
    # Maximum value for normalization (adjust as needed)
    max_val = max(np.max(np.abs(matrix)), 3.0)
    
    # Create the image pixel by pixel in HSV space
    for i in range(rows):
        for j in range(cols):
            value = matrix[i, j]
            
            if value == 0:
                # Zero values are black (set V=0)
                hsv_image[i, j] = [0, 0, 0]  # Any hue, no saturation, no value = black
            elif value > 0:
                # Positive values are blue (hue = 0.6)
                # Magnitude determines both saturation and value
                magnitude = min(1.0, value / max_val)
                hsv_image[i, j] = [0.6, 1.0, 0.3 + 0.7 * magnitude]  # Blue with varying intensity
            else:
                # Negative values are red (hue = 1.0)
                # Magnitude determines both saturation and value
                magnitude = min(1.0, abs(value) / max_val)
                hsv_image[i, j] = [1.0, 1.0, 0.3 + 0.7 * magnitude]  # Red with varying intensity
    
    # Convert HSV to RGB for display
    rgb_image = hsv_to_rgb(hsv_image)
    
    # Create a figure with appropriate size
    plt.figure(figsize=(max(5, cols/2), max(5, rows/2)))
    
    # Display the image
    plt.imshow(rgb_image, interpolation='nearest')
    plt.grid(False)
    
    # Add column labels
    plt.xticks(range(cols), [f'Col {j}' for j in range(cols)])
    
    # Add row labels
    plt.yticks(range(rows), [f'Row {i}' for i in range(rows)])
    
    # Add a title
    plt.title('Matrix Visualization (Zero=Black, Positive=Blue, Negative=Red)')
    
    # Create custom legend patches
    legend_elements = [
        mpatches.Patch(color='black', label='Zero (0)'),
        mpatches.Patch(color=hsv_to_rgb([0.6, 1.0, 0.4]), label='Small Positive'),
        mpatches.Patch(color=hsv_to_rgb([0.6, 1.0, 0.7]), label='Medium Positive'),
        mpatches.Patch(color=hsv_to_rgb([0.6, 1.0, 1.0]), label='Large Positive'),
        mpatches.Patch(color=hsv_to_rgb([1.0, 1.0, 0.4]), label='Small Negative'),
        mpatches.Patch(color=hsv_to_rgb([1.0, 1.0, 0.7]), label='Medium Negative'),
        mpatches.Patch(color=hsv_to_rgb([1.0, 1.0, 1.0]), label='Large Negative')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add text annotations with values
    for i in range(rows):
        for j in range(cols):
            text_color = 'white' if matrix[i, j] == 0 or abs(matrix[i, j]) > max_val/3 else 'black'
            plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', 
                     color=text_color, fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return rgb_image  # Return the image for further processing if needed

# Example usage
if __name__ == "__main__":
    # Create an example matrix
    example_matrix = np.array([
        [2.5, 0.8, 0.0, 1.0],
        [0.3, 4.0, 1.0, -2.7],
        [1.0, -0.5, 0.7, 3.2],
        [5.1, 0.0, 0.2, -1.9],
        [0.5, 2.1, -3.0, 0.0]
    ])
    
    visualize_matrix_hsv(example_matrix)
    
    # Can also visualize a single vector
    example_vector = np.array([0.0, 1.0, 3.0, -2.0, 0.5])
    visualize_matrix_hsv(example_vector)
