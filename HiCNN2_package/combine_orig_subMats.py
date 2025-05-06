import numpy as np
import sys
import math

# 2025-05-05 AG: Script to join original submatrices

def main():
    if len(sys.argv) != 6:
        print("Usage: python join_orig_subMats.py <input_subMats> <input_indices> <chr_len> <resolution> <output_file>")
        sys.exit(1)

    # Parse arguments
    dat_predict = np.load(sys.argv[1])  # Original submatrices (40x40)
    dat_index = np.load(sys.argv[2])    # Indices
    chr_len = int(sys.argv[3])          # Chromosome length
    resolution = int(sys.argv[4])       # Resolution
    file_output = sys.argv[5]           # Output file

    # Calculate number of bins
    num_bins = math.ceil(chr_len / resolution)

    print("First few indices:", dat_index[:5])
    print("Number of bins:", num_bins)
    
    # Create empty matrix
    mat = np.zeros((num_bins, num_bins))

    # In the loop where placing submatrices
    for i in range(dat_predict.shape[0]):
        # Use indices directly without offset
        r1 = dat_index[i,0]
        c1 = dat_index[i,1]
        r2 = r1 + 40
        c2 = c1 + 40
        
        # Calculate valid region bounds
        r_start = max(0, r1)
        r_end = min(num_bins, r2)
        c_start = max(0, c1)
        c_end = min(num_bins, c2)
        
        # Skip if no valid region
        if r_start >= r_end or c_start >= c_end:
            continue
            
        # Calculate submatrix slice
        sub_r_start = 0  # Start from beginning of submatrix
        sub_r_end = r_end - r_start  # Number of rows to take
        sub_c_start = 0
        sub_c_end = c_end - c_start  # Number of columns to take
        
        # Debug print shapes
        print(f"Index {i}: Placing {dat_predict[i, sub_r_start:sub_r_end, sub_c_start:sub_c_end].shape} into {r_start}:{r_end}, {c_start}:{c_end}")
        print(f"Submatrix shape: {dat_predict[i, sub_r_start:sub_r_end, sub_c_start:sub_c_end].shape}")
        print(f"Target region shape: {(r_end - r_start, c_end - c_start)}")
        
        # Place the submatrix
        try:
            # testing the transpose
            mat[r_start:r_end, c_start:c_end] = np.squeeze(dat_predict[i, sub_r_start:sub_r_end, sub_c_start:sub_c_end]).T
        except ValueError as e:
            print(f"Error at index {i}: {e}")
            print(f"Matrix shape: {mat.shape}")
            print(f"Submatrix shape: {dat_predict[i].shape}")
            print(f"Target region: {r_start}:{r_end}, {c_start}:{c_end}")
            print(f"Submatrix slice: {sub_r_start}:{sub_r_end}, {sub_c_start}:{sub_c_end}")
            raise

    # Make the matrix symmetric
    lower_index = np.tril_indices(num_bins, -1)
    mat[lower_index] = mat.T[lower_index]

    # Save the result
    np.save(file_output, mat)

if __name__ == '__main__':
    main()
