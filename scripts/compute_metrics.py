import cooler
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def check_cool_compatibility(cool1, cool2):
    clr1 = cooler.Cooler(cool1)
    clr2 = cooler.Cooler(cool2)

    bin_size1 = clr1.binsize
    bin_size2 = clr2.binsize

    shape1 = clr1.shape
    shape2 = clr2.shape

    compatible = (bin_size1 == bin_size2) and (shape1 == shape2)

    return {
        "compatible": compatible,
        "binsize_match": bin_size1 == bin_size2,
        "shape_match": shape1 == shape2,
        "binsize1": bin_size1,
        "binsize2": bin_size2,
        "shape1": shape1,
        "shape2": shape2,
    }

def load_contact_matrix(cool_path):
    clr = cooler.Cooler(cool_path)
    matrix = clr.matrix(balance=True)[:]
    matrix = np.nan_to_num(matrix)
    return matrix

def compute_metrics(mat1, mat2):
    flat1, flat2 = mat1.flatten(), mat2.flatten()

    # In a balanced matrix, values are in the range [0, 1].
    data_range = 1

    metrics = {
        "Pearson Correlation (High=Better)": pearsonr(flat1, flat2)[0],
        "Spearman Correlation (High=Better)": spearmanr(flat1, flat2)[0],
        "PSNR (High=Better)": psnr(mat1, mat2, data_range=data_range),
        "SSIM (High=Better)": ssim(mat1, mat2, data_range=data_range),
        "MSE (Low=Better)": mean_squared_error(flat1, flat2),
        "MAE (Low=Better)": mean_absolute_error(flat1, flat2),
    }
    return metrics

def compare_cool_files(cool_file1, cool_file2):
    result = check_cool_compatibility(cool_file1, cool_file2)

    if not result["compatible"]:
        raise ValueError(
            f"You provided incompatible .cool files.\n"
            f"  Bin sizes are {result['binsize1']} and {result['binsize2']}.\n"
            f"  Matrix shapes are {result['shape1']} and {result['shape2']}."
        )

    mat1 = load_contact_matrix(cool_file1)
    mat2 = load_contact_matrix(cool_file2)

    metrics = compute_metrics(mat1, mat2)
    return metrics

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compare two .cool files using similarity metrics.")
    parser.add_argument("cool_file1", help="Provide path to the first .cool file.")
    parser.add_argument("cool_file2", help="Provide path to the second .cool file.")
    args = parser.parse_args()

    try:
        results = compare_cool_files(args.cool_file1, args.cool_file2)
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
    except Exception as e:
        print(f"Comparison failed due to {e}.")
        sys.exit(1)
