import numpy as np
import sys

def main():
    if len(sys.argv) != 5:
        sys.stderr.write("Usage: python mark_ROIs.py <submats.npy> <indices.npy> <chr_len> <resolution> <output_prefix>\n")
        sys.exit(1)
    
    # Load input files
    sub_mats = np.load(sys.argv[1])
    indices = np.load(sys.argv[2])
    resolution = int(sys.argv[3])
    output_prefix = sys.argv[4]
    #method = sys.argv[5]
    
    # Initialize ROI array (1=ROI, 0=non-ROI)
    roi_flags = np.zeros(len(indices), dtype=bool)
    
    # Mark random submatrices as ROIs
    np.random.seed(42)

    diag_idx = np.where(indices[:,0] == indices[:,1])[0]
    np.random.shuffle(diag_idx)
    roi_flags[diag_idx[:int(0.1*len(diag_idx))]] = True

    #print("roi_flags: ", roi_flags)
    #print("roi_flags number: ", np.sum(roi_flags))

    # Mark diagonal-adjacent submatrices as ROIs
    for i in range(len(diag_idx)):
        if roi_flags[diag_idx[i]]:
            col = np.where(indices[:,1] == indices[diag_idx[i],0])[0]
            row = np.where(indices[:,0] == indices[diag_idx[i],1])[0]
            roi_flags[col] = True
            roi_flags[row] = True

    # Save ROI indices
    roi_indices = np.where(roi_flags)[0]
    print("number of roi_indices: ", len(roi_indices))
    print("number of non roi indices: ", len(np.where(~roi_flags)[0]))
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

if __name__ == "__main__":
    main()