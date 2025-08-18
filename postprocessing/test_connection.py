# import nibabel as nib
# import numpy as np
# from medpy.metric.binary import dc, assd
# from skimage.measure import label

# def dice_score(pred, gt):
#     return dc(pred, gt)

# def assd_score(pred, gt, voxelspacing=None):
#     try:
#         return assd(pred, gt, voxelspacing=voxelspacing)
#     except RuntimeError:
#         return np.nan  

# def connectivity_error_ratio(pred, gt):

#     beta0_pred = label(pred, connectivity=3).max()
#     beta0_gt = label(gt, connectivity=3).max()
#     if beta0_gt == 0:
#         return np.nan
#     return abs(beta0_pred - beta0_gt) / beta0_gt

# def load_mask(path):
#     img = nib.load(path)
#     data = img.get_fdata()
#     return (data > 0).astype(np.uint8), img.header.get_zooms()  

# def evaluate(pred_path, gt_path, name=""):
#     pred_mask, spacing = load_mask(pred_path)
#     gt_mask, _ = load_mask(gt_path)

#     dsc = dice_score(pred_mask, gt_mask)
#     avg_assd = assd_score(pred_mask, gt_mask, voxelspacing=spacing)
#     conn_err = connectivity_error_ratio(pred_mask, gt_mask)

#     print(f"\n=== {name} 评估结果 ===")
#     print(f"Dice (DSC): {dsc:.4f}")
#     print(f"ASSD: {avg_assd:.4f} mm")
#     print(f"ϵβ₀ (连接性误差比): {conn_err:.4f}")

# if __name__ == "__main__":
#     # 修改为你的文件路径
#     gt_path = "COPDG13_GT_corrected.nii.gz"          # 标注（真值）
#     before_path = "COPDG13_training_result.nii.gz"  # 连接前结果
#     after_path = "final_connected_seg.nii.gz"    # 连接后结果

#     evaluate(before_path, gt_path, name="连接前")
#     evaluate(after_path, gt_path, name="连接后")

import nibabel as nib
import numpy as np
from medpy.metric.binary import dc, assd
from skimage.measure import label

def dice_score(pred, gt):
    return dc(pred, gt)

def assd_score(pred, gt, voxelspacing=None):
    try:
        return assd(pred, gt, voxelspacing=voxelspacing)
    except RuntimeError:
        return np.nan  

def connectivity_error_ratio(pred, gt):
    beta0_pred = label(pred, connectivity=3).max()
    beta0_gt = label(gt, connectivity=3).max()
    if beta0_gt == 0:
        return np.nan
    return abs(beta0_pred - beta0_gt) / beta0_gt

def load_mask(path, target_labels=None):
    img = nib.load(path)
    data = img.get_fdata()
    if target_labels is None:
        return data.astype(np.int32), img.header.get_zooms()
    else:
        mask = np.isin(data, target_labels).astype(np.uint8)
        return mask, img.header.get_zooms()

def evaluate_multilabel(pred_path, gt_path, name="", target_labels=None):
    if target_labels is not None:
        pred_mask, spacing = load_mask(pred_path, target_labels)
        gt_mask, _ = load_mask(gt_path, target_labels)
        dsc = dice_score(pred_mask, gt_mask)
        avg_assd = assd_score(pred_mask, gt_mask, voxelspacing=spacing)
        conn_err = connectivity_error_ratio(pred_mask, gt_mask)
        print(f"\n=== {name} 标签 {target_labels} 评估结果 ===")
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
            print(f"\n=== {name} 类别 {lbl} 评估结果 ===")
            print(f"Dice (DSC): {dsc:.4f}")
            print(f"ASSD: {avg_assd:.4f} mm")
            print(f"ϵβ₀: {conn_err:.4f}")


def summary(name, arr):
    import numpy as np
    print(f"[{name}] shape={arr.shape} voxels(A)={(arr==2).sum()} voxels(V)={(arr==3).sum()} uniq={np.unique(arr)[:8]}")


if __name__ == "__main__":   

    gt_path = "COPDG13_GT_corrected.nii.gz"          
    before_path = "COPDG13_training_result.nii.gz"  
    after_path = "vessel_reconnected.nii.gz" 
    # after_path = "COPDG13_training_result_reconnected.nii.gz" 

    pred = nib.load(before_path).get_fdata()
    gt   = nib.load(gt_path).get_fdata()
    after = nib.load(after_path).get_fdata()

    # print('pred shape:', pred.shape, 'uniq:', np.unique(pred).tolist()[:10])
    # print('gt   shape:', gt.shape,   'uniq:', np.unique(gt).tolist()[:10])
    # print('pred voxels of {2,3}:', int(((pred==2)|(pred==3)).sum()))
    # print('gt   voxels of {2,3}:', int(((gt==2)|(gt==3)).sum()))
    summary("GT", gt)
    summary("Before", pred)
    summary("After", after)



    labels_to_evaluate = [2, 3]
    evaluate_multilabel(before_path, gt_path, name="连接前", target_labels=labels_to_evaluate)
    evaluate_multilabel(after_path, gt_path, name="连接后", target_labels=labels_to_evaluate)

   
    # evaluate_multilabel(before_path, gt_path, name="连接前", target_labels=None)
    # evaluate_multilabel(after_path, gt_path, name="连接后", target_labels=None)final_connected_seg.nii.gz
