import numpy as np
import sys
import cooler
import pandas as pd
from collections import defaultdict
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert .npy contact matrix to .cool format')
    parser.add_argument('input', help='Input .npy matrix file')
    parser.add_argument('output', help='Output .cool file')
    parser.add_argument('--chrom', default='chr1', help='Chromosome name (default: chr1)')
    parser.add_argument('--resolution', type=int, default=10000, 
                       help='Resolution in base pairs (default: 10000)')
    args = parser.parse_args()

    try:
        # Load and clean the numpy matrix
        matrix = np.load(args.input)
        matrix = np.nan_to_num(matrix, posinf=1e6, neginf=-1e6)
        
        # Create bins (assuming square matrix)
        n = matrix.shape[0]
        bins = pd.DataFrame({
            'chrom': [args.chrom] * n,
            'start': np.arange(0, n * args.resolution, args.resolution),
            'end': np.arange(args.resolution, (n + 1) * args.resolution, args.resolution)
        })
        
        # Create pixel dataframe with duplicate handling
        rows, cols = np.where(matrix > 0)
        # Swap rows and cols where row > col to ensure bin1_id <= bin2_id
        swap_mask = rows > cols
        rows[swap_mask], cols[swap_mask] = cols[swap_mask], rows[swap_mask]
        
        # Aggregate counts for duplicate pixels
        pixel_counts = defaultdict(float)
        for r, c, v in zip(rows, cols, matrix[rows, cols]):
            pixel_counts[(r, c)] += v
        
        # Create final pixel dataframe
        rows, cols = zip(*pixel_counts.keys())
        pixels = pd.DataFrame({
            'bin1_id': rows,
            'bin2_id': cols,
            'count': list(pixel_counts.values())
        })
        
        # Create cooler file
        cooler.create_cooler(args.output, bins, pixels)
        print(f"Successfully created cooler file: {args.output}")
        
    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
