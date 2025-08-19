import nibabel as nib
import numpy as np
from medpy.metric.binary import dc, assd
from skimage.measure import label

def dice_score(pred, gt):
    """Compute Dice Similarity Coefficient (DSC)."""
    return dc(pred, gt)

def assd_score(pred, gt, voxelspacing=None):
    """Compute Average Symmetric Surface Distance (ASSD)."""
    try:
        return assd(pred, gt, voxelspacing=voxelspacing)
    except RuntimeError:
        return np.nan  

def connectivity_error_ratio(pred, gt):
    """Compute connectivity error ratio (ϵβ₀)."""
    beta0_pred = label(pred, connectivity=3).max()
    beta0_gt = label(gt, connectivity=3).max()
    if beta0_gt == 0:
        return np.nan
    return abs(beta0_pred - beta0_gt) / beta0_gt

def load_mask(path, target_labels=None):
    """Load NIfTI mask. Return full mask or binary mask for given labels."""
    img = nib.load(path)
    data = img.get_fdata()
    if target_labels is None:
        return data.astype(np.int32), img.header.get_zooms()
    else:
        mask = np.isin(data, target_labels).astype(np.uint8)
        return mask, img.header.get_zooms()

def evaluate_multilabel(pred_path, gt_path, name="", target_labels=None):
    """Evaluate segmentation with DSC, ASSD, and connectivity error."""
    if target_labels is not None:
        pred_mask, spacing = load_mask(pred_path, target_labels)
        gt_mask, _ = load_mask(gt_path, target_labels)
        dsc = dice_score(pred_mask, gt_mask)
        avg_assd = assd_score(pred_mask, gt_mask, voxelspacing=spacing)
        conn_err = connectivity_error_ratio(pred_mask, gt_mask)
        print(f"\n=== {name} label {target_labels} Evaluation result ===")
        print(f"Dice (DSC): {dsc:.4f}")
        print(f"ASSD: {avg_assd:.4f} mm")
        print(f"ϵβ₀: {conn_err:.4f}")
    else:
        data_gt, _ = load_mask(gt_path)
        all_labels = sorted(set(np.unique(data_gt)) - {0})
        for lbl in all_labels:
            pred_mask, spacing = load_mask(pred_path, [lbl])
            gt_mask, _ = load_mask(gt_path, [lbl])
            dsc = dice_score(pred_mask, gt_mask)
            avg_assd = assd_score(pred_mask, gt_mask, voxelspacing=spacing)
            conn_err = connectivity_error_ratio(pred_mask, gt_mask)
            print(f"\n=== {name} label {lbl} Evaluation result ===")
            print(f"Dice (DSC): {dsc:.4f}")
            print(f"ASSD: {avg_assd:.4f} mm")
            print(f"ϵβ₀: {conn_err:.4f}")


def summary(name, arr):
    """Print array stats: shape, voxel counts, unique labels."""
    print(f"[{name}] shape={arr.shape} voxels(A)={(arr==2).sum()} voxels(V)={(arr==3).sum()} uniq={np.unique(arr)[:8]}")


if __name__ == "__main__":   

    gt_path = "Ground_truth.nii.gz"          
    before_path = "input_vessel.nii.gz"  
    after_path = "vessel_reconnected.nii.gz" 
    

    pred = nib.load(before_path).get_fdata()
    gt   = nib.load(gt_path).get_fdata()
    after = nib.load(after_path).get_fdata()

    summary("GT", gt)
    summary("Before", pred)
    summary("After", after)


    # test target class
    labels_to_evaluate = [2, 3]
    evaluate_multilabel(before_path, gt_path, name="before connecting", target_labels=labels_to_evaluate)
    evaluate_multilabel(after_path, gt_path, name="after connecting", target_labels=labels_to_evaluate)

    # test all classes
    # evaluate_multilabel(before_path, gt_path, name="before connecting", target_labels=None)
    # evaluate_multilabel(after_path, gt_path, name="after connecting", target_labels=None)