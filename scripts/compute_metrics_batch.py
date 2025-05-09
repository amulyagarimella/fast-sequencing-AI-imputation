import os
import json
import warnings
import numpy as np
import cooler
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

warnings.filterwarnings('ignore')

def check_cool_compatibility(cool1, cool2, chr_num=None):
    """Check if two .cool files have compatible dimensions"""
    clr1 = cooler.Cooler(cool1)
    clr2 = cooler.Cooler(cool2)
    
    if chr_num:
        chr_str = f"chr{chr_num}"
        try:
            bins1 = clr1.bins().fetch(chr_str)
            bins2 = clr2.bins().fetch(chr_str)
            shape1 = len(bins1)
            shape2 = len(bins2)
            bin_size1 = clr1.binsize
            bin_size2 = clr2.binsize
        except ValueError as e:
            return {
                "compatible": False,
                "error": f"Chromosome {chr_num} missing in one file: {str(e)}",
                "binsize1": clr1.binsize,
                "binsize2": clr2.binsize,
                "shape1": clr1.shape,
                "shape2": clr2.shape,
                "chromosome": chr_str
            }
    else:
        shape1 = clr1.shape
        shape2 = clr2.shape
        bin_size1 = clr1.binsize
        bin_size2 = clr2.binsize
    
    compatible = (bin_size1 == bin_size2) and (shape1 == shape2)
    
    return {
        "compatible": compatible,
        "binsize_match": bin_size1 == bin_size2,
        "shape_match": shape1 == shape2,
        "binsize1": bin_size1,
        "binsize2": bin_size2,
        "shape1": shape1,
        "shape2": shape2,
        "chromosome": f"chr{chr_num}" if chr_num else "whole_genome"
    }

def compute_metrics(mat1, mat2):
    """Compute all metrics on full matrices"""
    flat1, flat2 = mat1.flatten(), mat2.flatten()
    
    # Filter out NaN values
    mask = ~(np.isnan(flat1) | np.isnan(flat2))
    flat1 = flat1[mask]
    flat2 = flat2[mask]
    
    metrics = {
        "Pearson Correlation": pearsonr(flat1, flat2)[0],
        "Spearman Correlation": spearmanr(flat1, flat2)[0],
        "PSNR": psnr(mat1, mat2, data_range=1),
        "SSIM": ssim(mat1, mat2, data_range=1),
        "MSE": mean_squared_error(flat1, flat2),
        "MAE": mean_absolute_error(flat1, flat2)
    }
    return metrics

def run_comparison(cool1_path, cool2_pattern, output_path, chr_num=None):
    """Batch comparison using full matrices"""
    import glob
    import gc
    
    # Load base matrix ONCE
    print(f"Loading base matrix: {cool1_path}")
    clr1 = cooler.Cooler(cool1_path)
    mat1 = np.nan_to_num(
        clr1.matrix(balance=False, sparse=False).fetch(f"chr{chr_num}") 
        if chr_num else 
        clr1.matrix(balance=False, sparse=False)[:]
    )
    print(f"Base matrix shape: {mat1.shape} ({mat1.nbytes/1e6:.1f}MB)")
    
    # Process all comparison files
    results = {"base_file": cool1_path, "comparisons": {}}
    cool2_files = sorted(glob.glob(cool2_pattern))
    
    for cool2_path in tqdm(cool2_files, desc="Processing comparisons"):
        try:
            # Load comparison matrix
            clr2 = cooler.Cooler(cool2_path)
            mat2 = np.nan_to_num(
                clr2.matrix(balance=False, sparse=False).fetch(f"chr{chr_num}") 
                if chr_num else 
                clr2.matrix(balance=False, sparse=False)[:]
            )
            
            if mat1.shape != mat2.shape:
                raise ValueError(f"Shape mismatch: {mat1.shape} vs {mat2.shape}")
                
            # Compute and store metrics
            results["comparisons"][cool2_path] = compute_metrics(mat1, mat2)
            
            # Explicit cleanup
            del clr2, mat2
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {cool2_path}: {str(e)}")
            results["comparisons"][cool2_path] = {"error": str(e)}
            if 'clr2' in locals():
                del clr2
            if 'mat2' in locals():
                del mat2
            gc.collect()
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare .cool files using contact matrix similarity metrics.")
    parser.add_argument("--cool1", help="Provide path to base .cool file.")
    parser.add_argument("--cool2", help="Provide glob pattern for comparison .cool files.")
    parser.add_argument("--output", help="Provide path to output JSON metrics file.")
    parser.add_argument("--chr", type=int, help="Chromosome number to process (e.g., 21 for chr21). If not specified, processes entire genome.")

    args = parser.parse_args()
    run_comparison(args.cool1, args.cool2, args.output, args.chr)
