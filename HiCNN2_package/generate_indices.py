import numpy as np
import sys
import math

# 2025-05-05 AG: Generate indices for HiCNN2 submatrices
def generate_indices(chr_len, resolution, sub_mat_size, step_size, output_file):
    """
    Generate indices for submatrices in a Hi-C matrix
    
    Args:
        chr_len: Length of the chromosome in base pairs
        resolution: Resolution of the Hi-C matrix (e.g., 10kb)
        sub_mat_size: Size of the submatrices (40 for original, 28 for predictions)
        step_size: Step size for sliding window (28 for original, 28 for predictions)
        output_file: Output file for indices
    """
    num_bins = math.ceil(chr_len / resolution)
    
    # Generate indices
    indices = []
    for i in range(0, num_bins, step_size):
        for j in range(i, num_bins, step_size):
            # Add offset for original submatrices
            offset = (40 - sub_mat_size) // 2
            indices.append((i + offset, j + offset))
    
    indices = np.array(indices)
    np.save(output_file, indices)
    
    print(f"Generated {len(indices)} indices")
    print(f"First index: {indices[0]}")
    print(f"Last index: {indices[-1]}")
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python generate_indices.py <chr_len> <resolution> <sub_mat_size> <step_size> <output_file>")
        print("Example for original submatrices: python generate_indices.py 249250621 10000 40 28 indices_orig.npy")
        print("Example for prediction submatrices: python generate_indices.py 249250621 10000 28 28 indices_pred.npy")
        sys.exit(1)

    chr_len = int(sys.argv[1])
    resolution = int(sys.argv[2])
    sub_mat_size = int(sys.argv[3])
    step_size = int(sys.argv[4])
    output_file = sys.argv[5]

    generate_indices(chr_len, resolution, sub_mat_size, step_size, output_file)
