import cooler
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python get_chr_sizes.py <input.mcool> <resolution> <output.sizes>")
        sys.exit(1)
    
    clr = cooler.Cooler(sys.argv[1] + '::resolutions/' + sys.argv[2])
    with open(sys.argv[3], 'w') as f:
        for chrom in clr.chromsizes.index:
            f.write(f"{chrom}\t{clr.chromsizes[chrom]}\n")

if __name__ == "__main__":
    main()