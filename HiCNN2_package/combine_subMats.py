
import sys
import numpy as np

# 2025-05-05 AG: Added sub_mat_n parameter

dat_predict = np.squeeze(np.load(sys.argv[1]).astype(np.float32))
dat_index   = np.load(sys.argv[2])
chr_len     = int(sys.argv[3])
resolution  = int(sys.argv[4])
file_output = sys.argv[5]

if len(sys.argv) > 6:
	sub_mat_n = int(sys.argv[6])
else:
	sub_mat_n = 28

num_bins = np.ceil(chr_len / resolution).astype('int')
mat = np.zeros((num_bins, num_bins))

for i in range(dat_predict.shape[0]):
	r1 = dat_index[i,0]
	c1 = dat_index[i,1]
	r2 = r1 + sub_mat_n
	c2 = c1 + sub_mat_n
	try:
		mat[r1:r2, c1:c2] = dat_predict[i,:,:]
	except:
		print("Error at index:", i)
		continue


# copy upper triangle to lower triangle
lower_index = np.tril_indices(num_bins, -1)
mat[lower_index] = mat.T[lower_index]  


np.save(file_output, mat)


