import cooler
from cooler import balance_cooler
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tempfile
import os
import json

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

def load_contact_matrix(cool_path, chr_num):
    clr = cooler.Cooler(cool_path)
    balance_cooler(clr, store=True)
    matrix = clr.matrix(balance=True).fetch(f"chr{chr_num}")
    return np.nan_to_num(matrix)

def compute_metrics(mat1, mat2):
    flat1, flat2 = mat1.flatten(), mat2.flatten()

    # Balanced values should range in [0, 1].
    data_range = 1
    return {
        "Pearson Correlation (High=Better)": pearsonr(flat1, flat2)[0],
        "Spearman Correlation (High=Better)": spearmanr(flat1, flat2)[0],
        "PSNR (High=Better)": psnr(mat1, mat2, data_range=data_range),
        "SSIM (High=Better)": ssim(mat1, mat2, data_range=data_range),
        "MSE (Low=Better)": mean_squared_error(flat1, flat2),
        "MAE (Low=Better)": mean_absolute_error(flat1, flat2),
    }

def compare_cool_files(cool_file1, cool_file2, chr_num):
    result = check_cool_compatibility(cool_file1, cool_file2)
    if not result["compatible"]:
        raise ValueError(
            f"You provided incompatible .cool files.\n"
            f"  Bin sizes are {result['binsize1']} and {result['binsize2']}.\n"
            f"  Matrix shapes are {result['shape1']} and {result['shape2']}."
        )
    mat1 = load_contact_matrix(cool_file1, chr_num)
    mat2 = load_contact_matrix(cool_file2, chr_num)
    return compute_metrics(mat1, mat2)

def run_test(cool1_path=None, cool2_path=None, output_metrics_path=None, chr_num=None):
    if cool1_path and cool2_path and output_metrics_path:
        results = compare_cool_files(cool1_path, cool2_path, chr_num)
        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        with open(output_metrics_path, "w") as f:
            json.dump(results, f, indent=4)
    elif cool1_path is None and cool2_path is None and output_metrics_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_metrics_path = os.path.join(script_dir, "..", "outputs", "compute_metrics_test.json")
        input_mcool_path = os.path.join(script_dir, "..", "data", "GM12878.GSE115524.Homo_Sapiens.CTCF.b1.mcool")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_cool1 = os.path.join(tmpdir, "chr21_5kb_1.cool")
            tmp_cool2 = os.path.join(tmpdir, "chr21_5kb_2.cool")

            for output_cool in [tmp_cool1, tmp_cool2]:
                clr = cooler.Cooler(f"{input_mcool_path}::resolutions/5000")
                chr21_bins = clr.bins().fetch("chr21")
                chr21_bin_ids = chr21_bins.index.values
                bin_id_map = {old_id: new_id for new_id, old_id in enumerate(chr21_bin_ids)}

                pixels_df = clr.pixels()[:]
                mask = pixels_df["bin1_id"].isin(chr21_bin_ids) & pixels_df["bin2_id"].isin(chr21_bin_ids)
                pixels_chr21 = pixels_df[mask].copy()
                pixels_chr21["bin1_id"] = pixels_chr21["bin1_id"].map(bin_id_map)
                pixels_chr21["bin2_id"] = pixels_chr21["bin2_id"].map(bin_id_map)

                cooler.create_cooler(
                    output_cool,
                    bins=chr21_bins,
                    pixels=pixels_chr21,
                    assembly=clr.info.get("genome-assembly", None),
                    ordered=True
                )

            results = compare_cool_files(tmp_cool1, tmp_cool2, chr_num)

        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        with open(output_metrics_path, "w") as f:
            json.dump(results, f, indent=4)
    else:
        raise ValueError(
            f"Please enter arguments --cool1 <path> --cool2 <path> --output <path> or nothing at all to test."
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare .cool files using contact matrix similarity metrics.")
    parser.add_argument("--cool1", help="Provide path to first .cool file.")
    parser.add_argument("--cool2", help="Provide path to second .cool file.")
    parser.add_argument("--output", help="Provide path to output JSON metrics file.")
    parser.add_argument("--chr-num", type=int, help="Provide chromosome number to analyze.")

    args = parser.parse_args()

    run_test(args.cool1, args.cool2, args.output, args.chr_num)