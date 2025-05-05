from imaplib import ParseFlags
import numpy as np
import sys
import joblib
import sklearn
from scipy import sparse

def main():
    if len(sys.argv) != 5:
        sys.stderr.write("Usage: python mark_ROIs.py <submats.npy> <indices.npy> <chr_len> <resolution> <output_prefix> <method - optional> <top_percent - optional>\n")
        sys.exit(1)
    
    # Load input files
    sub_mats = np.load(sys.argv[1])
    indices = np.load(sys.argv[2])
    resolution = int(sys.argv[3])
    output_prefix = sys.argv[4]

    if len(sys.argv) >= 6:
        method = sys.argv[5]
    else:
        method = "ridge"
    
    if len(sys.argv) == 7:
        top_percent = float(sys.argv[6])/100
    else:
        top_percent = None

    if top_percent is None and method == "ridge":
        top_percent = 0.1
    
    # Initialize ROI array (1=ROI, 0=non-ROI)
    roi_flags = np.zeros(len(indices), dtype=bool)
    
    # Mark random submatrices as ROIs
    np.random.seed(42)

    diag_idx = np.where(indices[:,0] == indices[:,1])[0]
    if method == "random":
        np.random.shuffle(diag_idx)
        roi_flags[diag_idx[:int(top_percent*len(diag_idx))]] = True
    elif method == "logreg":
        raise NotImplementedError
    elif method == "ridge": # ridge regression
        model = joblib.load("prediction_models/ridge_model_csc_40.joblib")
        features_csc = sparse.vstack([sparse.csc_matrix(np.squeeze(sub_mats[i]).flatten()) for i in diag_idx], format='csc')
        print(features_csc.shape)
        preds = model.predict(features_csc)
        # ROIs are top n predicted
        top_n = int(top_percent*len(diag_idx))
        top_idx = np.argsort(preds)[::-1][:top_n]
        roi_flags[diag_idx[top_idx]] = True
    else:
        raise ValueError("Invalid method: " + method)

    # Mark diagonal-adjacent submatrices as ROIs
    for i in range(len(diag_idx)):
        if roi_flags[diag_idx[i]]:
            col = np.where(indices[:,1] == indices[diag_idx[i],0])[0]
            row = np.where(indices[:,0] == indices[diag_idx[i],1])[0]
            roi_flags[col] = True
            roi_flags[row] = True

    # Save ROI indices
    roi_indices = np.where(roi_flags)[0]
    #print("number of roi_indices: ", len(roi_indices))
    #print("number of non roi indices: ", len(np.where(~roi_flags)[0]))
    np.save(output_prefix + '.npy', roi_flags)
    
    # Save bed if needed
    with open(output_prefix + '.bed', 'w') as f:
        for idx in roi_indices:
            bin1, bin2 = indices[idx]
            start1 = bin1 * resolution
            end1 = (bin1 + 1) * resolution
            start2 = bin2 * resolution
            end2 = (bin2 + 1) * resolution
            f.write(f"chrN\t{start1}\t{end1}\tchrN:{start2}-{end2}\n")
        
    with open(output_prefix + '.txt', 'w') as f:
        f.write(f"Num ROIs: {len(roi_indices)} out of {len(indices)}\n")
        f.write(f"Method used to calculate ROIs: {method}\n")
        f.write(f"Top percent passed in: {top_percent}\n")

if __name__ == "__main__":
    main()