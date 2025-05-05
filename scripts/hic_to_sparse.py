import numpy as np
import sys
from collections import defaultdict

def main():
    if len(sys.argv) != 2:
        print("Usage: python hic_to_sparse.py <input.npy>")
        sys.exit(1)
    
    mat = np.load(sys.argv[1])
    pixel_counts = defaultdict(float)
    
    rows, cols = np.nonzero(mat)
    data = mat[rows, cols]
    
    # Aggregate counts for duplicate pixels
    for r, c, v in zip(rows, cols, data):
        print("Duplicate pixel found at (", r, ",", c, ") with value", v)
        pixel_counts[(r, c)] += v
    
    # Output unique pixels
    for (r, c), v in pixel_counts.items():
        print(f"{r}\t{c}\t{v}")

if __name__ == "__main__":
    main()