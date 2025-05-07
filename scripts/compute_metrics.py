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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def check_cool_compatibility(cool1, cool2, chr_num=None):
    if chr_num:
        clr1 = cooler.Cooler(cool1)
        clr2 = cooler.Cooler(cool2)
        bin_size1 = clr1.binsize
        bin_size2 = clr2.binsize
        shape1 = clr1.bins().fetch(f"chr{chr_num}").shape[0]
        shape2 = clr2.bins().fetch(f"chr{chr_num}").shape[0]
    else:
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

def load_contact_matrix_streaming(cool_path, chunk_size=100, chr_num=None, super_chunk_size=1000):
    clr = cooler.Cooler(cool_path)
    balance_cooler(clr, store=True)
    
    # Get matrix dimensions
    if chr_num:
        chr_bins = clr.bins().fetch(f"chr{chr_num}")
        n_bins = len(chr_bins)
        print(f"Processing chromosome {chr_num} with {n_bins} bins")
    else:
        n_bins = clr.shape[0]
        print(f"Processing entire genome with {n_bins} bins")
    
    # Generator to yield chunks using super-chunks
    def chunk_generator():
        # Process in larger super-chunks
        for i_super in range(0, n_bins, super_chunk_size):
            i_super_end = min(i_super + super_chunk_size, n_bins)
            for j_super in range(0, n_bins, super_chunk_size):
                j_super_end = min(j_super + super_chunk_size, n_bins)
                
                # Load super-chunk once
                if chr_num:
                    super_chunk = np.nan_to_num(
                        clr.matrix(balance=True, sparse=False).fetch(f"chr{chr_num}")[
                            i_super:i_super_end, j_super:j_super_end
                        ]
                    )
                else:
                    super_chunk = np.nan_to_num(
                        clr.matrix(balance=True, sparse=False)[
                            i_super:i_super_end, j_super:j_super_end
                        ]
                    )
                
                # Process smaller chunks from super-chunk
                for i in range(i_super, i_super_end, chunk_size):
                    i_end = min(i + chunk_size, i_super_end)
                    for j in range(j_super, j_super_end, chunk_size):
                        j_end = min(j + chunk_size, j_super_end)
                        yield super_chunk[i-i_super:i_end-i_super, j-j_super:j_end-j_super]
    
    return chunk_generator()

def compute_metrics_streaming(chunk_gen1, chunk_gen2, n_bins, chunk_size):
    # Calculate total number of chunks
    total_chunks = (n_bins // chunk_size)**2
    if n_bins % chunk_size != 0:
        total_chunks += 2 * ((n_bins // chunk_size) + 1) - 1

    print(f"\nTotal chunks to process: {total_chunks}")
    print("Starting metric calculations...")
    
    # Initialize accumulators for streaming metrics
    pearson_sum = 0
    spearman_sum = 0
    psnr_sum = 0
    ssim_sum = 0
    mse_sum = 0
    mae_sum = 0
    num_chunks = 0
    num_corr_chunks = 0
    
    # Process chunks with progress bar
    with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as pbar:
        for chunk1, chunk2 in zip(chunk_gen1, chunk_gen2):
            flat1, flat2 = chunk1.flatten(), chunk2.flatten()
            
            # Check if either chunk is NaN
            if np.isnan(flat1).any() or np.isnan(flat2).any():
                continue
            
            # Check if either chunk is constant
            #if np.all(flat1 == flat1[0]) or np.all(flat2 == flat2[0]):
            #    continue
            
            num_chunks += 1
            num_corr_chunks += 1
            
            # Update accumulators

            if not (np.all(flat1 == flat1[0]) and np.all(flat2 == flat2[0])):
                pearson_sum += pearsonr(flat1, flat2)[0]
                spearman_sum += spearmanr(flat1, flat2)[0]
            psnr_sum += psnr(chunk1, chunk2, data_range=1)
            ssim_sum += ssim(chunk1, chunk2, data_range=1)
            mse_sum += mean_squared_error(flat1, flat2)
            mae_sum += mean_absolute_error(flat1, flat2)
            
            # Update progress bar
            pbar.update(1)
            
            # Discard chunks after processing
            del chunk1, chunk2
            
    # Calculate averages
    print("\nCalculating final metrics...")
    metrics = {}
    if num_chunks > 0:
        metrics = {
            "Pearson Correlation (High=Better)": pearson_sum / num_corr_chunks,
            "Spearman Correlation (High=Better)": spearman_sum / num_corr_chunks,
            "PSNR (High=Better)": psnr_sum / num_chunks,
            "SSIM (High=Better)": ssim_sum / num_chunks,
            "MSE (Low=Better)": mse_sum / num_chunks,
            "MAE (Low=Better)": mae_sum / num_chunks,
        }
    else:
        print("No valid chunks processed; metrics cannot be calculated.")
    
    print("\nFinal metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def compare_cool_files(cool_file1, cool_file2, chunk_size=100, super_chunk_size=1000, chr_num=None):
    print(f"\nStarting comparison of {cool_file1} and {cool_file2}")
    print(f"Using chunk size: {chunk_size}")
    if chr_num:
        print(f"Processing chromosome: chr{chr_num}")
    
    result = check_cool_compatibility(cool_file1, cool_file2, chr_num)
    if not result["compatible"]:
        raise ValueError(
            f"You provided incompatible .cool files.\n"
            f"  Bin sizes are {result['binsize1']} and {result['binsize2']}.\n"
            f"  Matrix shapes are {result['shape1']} and {result['shape2']}."
        )
    
    # Get matrix dimensions
    clr = cooler.Cooler(cool_file1)
    if chr_num:
        chr_bins = clr.bins().fetch(f"chr{chr_num}")
        n_bins = len(chr_bins)
        print(f"Chromosome {chr_num} dimensions: {n_bins} x {n_bins} bins")
    else:
        n_bins = clr.shape[0]
        print(f"Whole genome dimensions: {n_bins} x {n_bins} bins")
    print(f"Total chunks to process: {(n_bins // chunk_size)**2}")
    
    # Load and process chunks on the fly
    print("\nStarting chunk processing...")
    chunk_gen1 = load_contact_matrix_streaming(cool_file1, chunk_size, chr_num, super_chunk_size)
    chunk_gen2 = load_contact_matrix_streaming(cool_file2, chunk_size, chr_num, super_chunk_size)
    
    return compute_metrics_streaming(chunk_gen1, chunk_gen2, n_bins, chunk_size)

def run_test(cool1_path=None, cool2_path=None, output_metrics_path=None, chunk_size=100, super_chunk_size=1000, chr_num=None):
    if cool1_path and cool2_path and output_metrics_path:
        results = compare_cool_files(cool1_path, cool2_path, chunk_size=chunk_size, super_chunk_size=super_chunk_size, chr_num=chr_num)
        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        with open(output_metrics_path, "w") as f:
            json.dump(results, f, indent=4)
    elif cool1_path is None and cool2_path is None and output_metrics_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_metrics_path = os.path.join(script_dir, "..", "outputs", "compute_metrics_test.json")
        input_mcool_path = os.path.join(script_dir, "..", "data", "GM12878.GSE115524", "GM12878.GSE115524.Homo_Sapiens.CTCF.b1.mcool")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_cool1 = os.path.join(tmpdir, "chr21_5kb_1.cool")
            tmp_cool2 = os.path.join(tmpdir, "chr21_5kb_2.cool")

            for output_cool in [tmp_cool1, tmp_cool2]:
                clr = cooler.Cooler(f"{input_mcool_path}::resolutions/10000")
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

            results = compare_cool_files(tmp_cool1, tmp_cool2, chunk_size=chunk_size, super_chunk_size=super_chunk_size, chr_num=chr_num)

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
    parser.add_argument("--chunk-size", type=int, default=100, help="Size of chunks to process at a time (default: 100)")
    parser.add_argument("--super-chunk-size", type=int, default=1000, help="Size of super-chunks to load at once (default: 1000)")
    parser.add_argument("--chr", type=int, help="Chromosome number to process (e.g., 21 for chr21). If not specified, processes entire genome.")

    args = parser.parse_args()

    run_test(args.cool1, args.cool2, args.output, args.chunk_size, args.super_chunk_size, args.chr)
