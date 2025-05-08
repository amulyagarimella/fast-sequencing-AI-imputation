import sys
import numpy as np
import pickle
#import model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from model import model1
from model import model2
from model import model3

parser = argparse.ArgumentParser(description='HiCNN2 predicting process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-f1', '--file-test-data', type=str, metavar='FILE', required=True,
                        help='file name of the test data, npy format and shape=n1*1*40*40')
required.add_argument('-f2', '--file-test-predicted', type=str, metavar='FILE', required=True,
                        help='file name to save the predicted target, npy format and shape=n1*1*28*28')
required.add_argument('-mid', '--model', type=int, default=3, metavar='N', required=True,
                        help='1:HiCNN2-1, 2:HiCNN2-2, and 3:HiCNN2-3 (default: 3)')
required.add_argument('-m', '--file-best-model', type=str, metavar='FILE', required=True,
                        help='file name of the best model')
required.add_argument('-r', '--down-ratio', type=int, default=16, metavar='N', required=True,
                        help='down sampling ratio, 16 means 1/16 (default: 16)')
optional.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA predicting')
optional.add_argument('--HiC-max', type=int, default=100, metavar='N',
                        help='the maximum value of Hi-C contacts (default: 100)')
optional.add_argument('--resolution', type=int, default=10000, metavar='N',
                        help='resolution of Hi-C data (default: 10000)')
optional.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for test (default: 128)')
optional.add_argument('--roi-indices', type=str, metavar='FILE',
                        help='file name of the ROI indices, npy format and shape=n1')
optional.add_argument('--submat-indices', type=str, metavar='FILE',
                        help='file name of the submatrix indices, npy format')
optional.add_argument('--non-roi-method', type=str, default='expected', metavar='STR',
                        help='interpolation method for non-ROI regions (default: expected)')
optional.add_argument('--apply-down-ratio', action='store_true', default=False,
                        help='apply downscaling ratio to test data')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if args.model == 1:
	print("Using HiCNN2-1...")
	Net = model1.Net().to(device).eval()
elif args.model == 2:
	print("Using HiCNN2-2...")
	Net = model2.Net().to(device).eval()
else:
	print("Using HiCNN2-3...")
	Net = model3.Net().to(device).eval()

# AG 2025-05-04: edit to enable CPU usage
if use_cuda:
    Net.load_state_dict(torch.load(args.file_best_model))
else:
    Net.load_state_dict(torch.load(args.file_best_model, map_location=device))

if args.apply_down_ratio:
    low_res_test = np.minimum(args.HiC_max, np.load(args.file_test_data).astype(np.float32) * args.down_ratio)
else:
    low_res_test = np.load(args.file_test_data).astype(np.float32)

# Load submatrix indices if provided
submat_indices = None
if args.submat_indices is not None:
    submat_indices = np.load(args.submat_indices)

# Validate required parameters when ROIs are specified
if args.roi_indices is not None:
    if args.submat_indices is None:
        sys.stderr.write("Error: --submat-indices is required when using --roi-indices\n")
        sys.exit(1)
    if args.resolution is None:
        sys.stderr.write("Error: --resolution is required when using --roi-indices\n")
        sys.exit(1)
    if args.non_roi_method not in ['zero', 'lowres', 'expected']:
        sys.stderr.write("Error: --non-roi-method must be one of zero, lowres, or expected\n")
        sys.exit(1)

if args.roi_indices is not None:
    roi_indices = np.load(args.roi_indices)
    test_loader = torch.utils.data.DataLoader(data.TensorDataset(torch.from_numpy(low_res_test), torch.from_numpy(roi_indices)), batch_size=args.batch_size, shuffle=False)
else:
    test_loader = torch.utils.data.DataLoader(data.TensorDataset(torch.from_numpy(low_res_test), torch.from_numpy(np.ones(low_res_test.shape[0], dtype=int))), batch_size=args.batch_size, shuffle=False)


# Interpolation methods
def get_center_bins(non_roi, batch_start_idx):
    return np.squeeze(low_res_test[non_roi + batch_start_idx, :, 6:-6, 6:-6])

def expected_contacts(non_roi, batch_start_idx):
    bin1_start = submat_indices[non_roi + batch_start_idx, 0]
    bin2_start = submat_indices[non_roi + batch_start_idx, 1]
    offsets = np.arange(28)
    
    bin1 = bin1_start[:,None] + offsets[None,:]
    bin2 = bin2_start[:,None] + offsets[None,:]

    genomic_distance = np.abs(bin1[:,:,None] - bin2[:,None,:]) * args.resolution

    # Expected contacts calculation based on:
    # Lieberman-Aiden et al. (2009) Comprehensive mapping of long-range interactions
    # reveals folding principles of the human genome. Science, 326(5950):289-293.
    # Power-law exponent of -1.08 from Rao et al. (2014) Cell 159(7):1665-1680
    expected_contacts = 1 / (genomic_distance / args.resolution)**1.08
    mean_orig_contacts = low_res_test[non_roi + batch_start_idx,:,6:-6,6:-6].triu().mean(axis=(1,2))[:,None,None]
    print(mean_orig_contacts.shape)
    print(expected_contacts.shape)

    # NB pastis uses the mean of the original contacts to scale the expected contacts
    return expected_contacts * mean_orig_contacts

result = np.zeros((low_res_test.shape[0],1,28,28))
for i, (data, roi) in enumerate(test_loader):
    i1 = i * args.batch_size
    i2 = i1 + args.batch_size
    #print("i: ", i)
    #print("i1: ", i1)
    #print("i2: ", i2)
    if i == int(low_res_test.shape[0]/args.batch_size):
        i2 = low_res_test.shape[0]
    
    # Get ROI mask for current batch
    #if args.roi_indices is not None:
    batch_roi_mask = roi.numpy()
    #else:
    #    batch_roi_mask = np.ones(data.shape[0])
    
    #print("data shape: ", data.shape)
    #print("batch_roi_mask shape: ", batch_roi_mask.shape)

    data2 = Variable(data).to(device)

    # Only run prediction for ROI samples
    roi_indices_in_batch = np.arange(data.shape[0])
    if args.roi_indices is not None:
        roi_indices_in_batch = np.where(batch_roi_mask)[0]
    
    if args.roi_indices is not None and len(roi_indices_in_batch) > 0:
        roi_data = data2[roi_indices_in_batch]
        output = Net(roi_data)
        roi_result = output.cpu().data.numpy()
        roi_result = np.squeeze(roi_result)
        
        # Place ROI predictions in result array
        result[i1:i2,0,:,:][batch_roi_mask] = roi_result

        # interpolate non-ROI predictions
        non_roi_indices_in_batch = np.where(batch_roi_mask == 0)[0]
        if len(non_roi_indices_in_batch) > 0:
            # Compute expected hi-c contacts in non-ROI regions
            # Use actual genomic positions from indices
            if args.non_roi_method == 'expected':
                expected = expected_contacts(non_roi_indices_in_batch, i1)
                result[i1:i2,0,:,:][batch_roi_mask == 0] = expected
            elif args.non_roi_method == 'lowres':
                center_bins = get_center_bins(non_roi_indices_in_batch, i1)
                result[i1:i2,0,:,:][batch_roi_mask == 0] = center_bins
            elif args.non_roi_method == 'zero':
                continue
            else:
                raise ValueError(f"Unknown interpolation method: {args.non_roi_method}")
    else:
        output = Net(data2)
        resulti = output.cpu().data.numpy()
        resulti = np.squeeze(resulti)
        result[i1:i2,0,:,:] = resulti

np.save(args.file_test_predicted, result)
