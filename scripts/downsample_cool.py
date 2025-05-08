#!/usr/bin/env python
import cooler
import numpy as np
import random
import argparse
from tqdm import tqdm
import os

def downsample_cool(input_cool, output_cool, denominator, resolution=10000, chrom_num=None):
    """
    Downsample a .cool file using binomial sampling of contacts. 
    (inspired by HiCPlus - Zhang et al 2018 Nature, HiCNN - Liu et al 2019 Bioinformations, HiCNN2 - Liu et al 2019 Genes)
    """

    os.makedirs(os.path.dirname(output_cool), exist_ok=True)

    # Load the input cooler file
    c = cooler.Cooler(f"{input_cool}::resolutions/{resolution}")
    
    # Prepare pixel iterator
    if chrom_num:
        chrom = f"chr{chrom_num}"
        pixels = c.matrix(as_pixels=True, balance=False).fetch(chrom)
        bins = c.bins().fetch(chrom)
        #print(bins.head())
        bin_map = dict(zip(bins.index, np.arange(len(bins))))
        bins = bins.reset_index()
        #print(bins.head())
        #print(pixels.head())
        pixels["bin1_id"] = pixels["bin1_id"].map(bin_map)
        pixels["bin2_id"] = pixels["bin2_id"].map(bin_map)
        #print(pixels.head())
    else:
        pixels = c.pixels()[:]
        bins = c.bins()[:]

 
    # Apply binomial downsampling to counts
    pixels['count'] = np.random.binomial(n=pixels['count'], p=1/denominator)
    
    # Filter out zero-count pixels
    pixels = pixels[pixels['count'] > 0]

    # Aggregate duplicate pixels by summing their counts
    pixels = pixels.groupby(['bin1_id', 'bin2_id'], as_index=False)['count'].sum()
    
    #print(max(pixels['bin1_id']))
    #print(max(pixels['bin2_id']))
    #print(max(bins.index))
    # Create new cooler file
    cooler.create_cooler(
        output_cool,
        bins=bins,
        pixels=pixels,
        columns=['count'],
        dtypes={'count': int},
        assembly='hg38'
    )
    print(f"Downsampled cool file saved to {output_cool}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample Hi-C data from cool files')
    parser.add_argument('input_cool', help='Input cool file path')
    parser.add_argument('output_cool', help='Output cool file path')
    parser.add_argument('denominator', type=int, 
                       help='Downsampling ratio (e.g., 16 for 1/16 of reads)')
    parser.add_argument('--resolution', type=int, default=10000,
                       help='Resolution in base pairs (default: 10000)')
    parser.add_argument('--chrom-num', type=int, default=None,
                       help='Chromosome number to filter for (default: all)')
    
    args = parser.parse_args()
    
    downsample_cool(
        args.input_cool,
        args.output_cool,
        args.denominator,
        args.resolution,
        args.chrom_num
    )
