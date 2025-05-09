
import sys
import numpy as np
import os

# 2025-05-05 AG: Added sub_mat_n parameter

intermediates = np.load(sys.argv[1])
predictions = intermediates["predictions"]
diagonal = intermediates["is_diagonal"]
diagonal_adjacent = intermediates["is_diagonal_adjacent"]
dat_index   = intermediates["indices"]

chr_len     = int(sys.argv[2])
resolution  = int(sys.argv[3])

sub_mat_n = 28

output_dir = os.path.dirname(sys.argv[1])
predictions_output = os.path.join(output_dir, "predictions.mat")
diag_output = os.path.join(output_dir, "diagonal.mat")

num_bins = np.ceil(chr_len / resolution).astype('int')
predictions_mat = np.zeros((num_bins, num_bins))
diag_status_mat = np.zeros((num_bins, num_bins))

for i in range(predictions.shape[0]):
	r1 = dat_index[i,0]
	c1 = dat_index[i,1]
	r2 = r1 + sub_mat_n
	c2 = c1 + sub_mat_n
	try:
		predictions_mat[r1:r2, c1:c2] = predictions[i,6:-6,6:-6]
		diag_status_mat[r1:r2, c1:c2] = int(diagonal[i])
		if not diagonal[i]:
			diag_status_mat[r1:r2, c1:c2] = int(diagonal_adjacent[i])/2
	except Exception as e:
		print("Error at index:", i)
		print(e)
		continue


# copy upper triangle to lower triangle
np.save(predictions_output, predictions_mat)
np.save(diag_output, diag_status_mat)


