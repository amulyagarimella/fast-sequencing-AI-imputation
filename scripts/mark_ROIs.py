from imaplib import ParseFlags
import numpy as np
import sys
import joblib
import sklearn
from scipy import sparse
import argparse

def main():
    parser = argparse.ArgumentParser(description='Mark ROIs in Hi-C submatrices')
    parser.add_argument('submat_file', help='Input submatrices file (.npy)')
    parser.add_argument('index_file', help='Submatrix indices file (.npy)')
    parser.add_argument('resolution', type=int, help='Resolution in bp')
    parser.add_argument('output_prefix', help='Output prefix for ROI indices file')
    parser.add_argument('--sparsity', type=float, default=0.1, help='ROI sparsity threshold (default: 0.1)')
    parser.add_argument('--method', default='ridge', choices=['ridge', 'random', 'elasticnet', 'lasso', 'logistic'], 
                        help='ROI detection method (default: ridge)')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample factor (default: 1)')

    args = parser.parse_args()

    # Load input files
    sub_mats = np.load(args.submat_file)
    indices = np.load(args.index_file)

    # Initialize ROI array (1=ROI, 0=non-ROI)
    roi_flags = np.zeros(len(indices), dtype=bool)
    
    # Mark random submatrices as ROIs
    np.random.seed(42)

    diag_idx = np.where(indices[:,0] == indices[:,1])[0]
    features_csc = sparse.vstack([sparse.csc_matrix(np.squeeze(sub_mats[i]).flatten()) for i in diag_idx], format='csc')

    downsample_string=""
    if args.downsample > 1:
        downsample_string=f"_{args.downsample}down"

    if args.method == "random":
        np.random.shuffle(diag_idx)
        roi_flags[diag_idx[:int(args.sparsity*len(diag_idx))]] = True
    elif args.method == "ridge": # ridge regression
        model = joblib.load(f"prediction_models/ridge_model_csc_40{downsample_string}.joblib")
        preds = model.predict(features_csc)
        # ROIs are top n predicted
        top_n = int(args.sparsity*len(diag_idx))
        top_idx = np.argsort(preds)[::-1][:top_n]
        roi_flags[diag_idx[top_idx]] = True
    elif args.method == "elasticnet":
        model = joblib.load(f"prediction_models/elasticnet_model_csc_40{downsample_string}.joblib")
        preds = model.predict(features_csc)
        # ROIs are top n predicted
        top_n = int(args.sparsity*len(diag_idx))
        top_idx = np.argsort(preds)[::-1][:top_n]
        roi_flags[diag_idx[top_idx]] = True
    elif args.method == "lasso":
        model = joblib.load(f"prediction_models/lasso_model_csc_40{downsample_string}.joblib")
        preds = model.predict(features_csc)
        # ROIs are top n predicted
        top_n = int(args.sparsity*len(diag_idx))
        top_idx = np.argsort(preds)[::-1][:top_n]
        roi_flags[diag_idx[top_idx]] = True
    elif args.method == "logreg":
        model = joblib.load(f"prediction_models/logreg_model_csc_40{downsample_string}.joblib")
        preds = model.predict_proba(features_csc)[:,1]
        # ROIs are top n predicted
        top_n = int(args.sparsity*len(diag_idx))
        top_idx = np.argsort(preds)[::-1][:top_n]
        top_idx = top_idx[preds[top_idx] > 0.5]
        roi_flags[diag_idx[top_idx]] = True
    else:
        raise ValueError("Invalid method: " + args.method)

    # Mark diagonal-adjacent submatrices as ROIs
    for i in range(len(diag_idx)):
        if roi_flags[diag_idx[i]]:
            col = np.where(indices[:,1] == indices[diag_idx[i],0])[0]
            row = np.where(indices[:,0] == indices[diag_idx[i],1])[0]
            roi_flags[col] = True
            roi_flags[row] = True

    # Save ROI indices
    roi_indices = np.where(roi_flags)[0]
    np.save(args.output_prefix + '.npy', roi_flags)
    
    # Save bed if needed
    with open(args.output_prefix + '.bed', 'w') as f:
        for idx in roi_indices:
            bin1, bin2 = indices[idx]
            start1 = bin1 * args.resolution
            end1 = (bin1 + 1) * args.resolution
            start2 = bin2 * args.resolution
            end2 = (bin2 + 1) * args.resolution
            f.write(f"chrN\t{start1}\t{end1}\tchrN:{start2}-{end2}\n")
        
    with open(args.output_prefix + '.txt', 'w') as f:
        f.write(f"Num ROIs: {len(roi_indices)} of {len(indices)}\n")
        f.write(f"Final sparsity: {len(roi_indices)/len(indices)}\n")
        f.write(f"Method used to calculate ROIs: {args.method}\n")
        f.write(f"Sparsity threshold: {args.sparsity}\n")

if __name__ == "__main__":
    main()