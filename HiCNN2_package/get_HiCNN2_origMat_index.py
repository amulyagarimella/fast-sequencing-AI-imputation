
import numpy as np
import sys
import math

sub_mat_n = 40

# Human hg19 chromosome length
#chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566, 155270560]

# Mouse mm9 chromosome length
#chrs_length = [195471971,182113224,160039680,156508116,151834684,149736546,145441459,129401213,124595110,130694993,122082543,120129022,120421639,124902244,104043685,98207768,94987271,90702639,61431566,171031299]

chr_len          = int(sys.argv[1])
resolution       = int(sys.argv[2])
file_out_index   = sys.argv[3]


num_bins = math.ceil(chr_len / resolution)

index = []
for i in range(0, num_bins, sub_mat_n):
	for j in range(i, num_bins, sub_mat_n):
		index.append((i+6, j+6))

index = np.array(index)
np.save(file_out_index, index)


