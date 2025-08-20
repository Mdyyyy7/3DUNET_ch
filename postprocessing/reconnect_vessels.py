"""
reconnect_vessels.py

Implements post-processing of model prediction results.

The following functions are reused/adapted from:
- select_region (connectivity.py)
- perform_distance_trans (distance_transform.py)
- soft_dilate, soft_erode, soft_skel (get_skeleton.py)

Source project:
HiPaS_AV_Segmentation by Yuetan Chu
Repository: https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation
License: MIT License
"""
import argparse
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage.measure import label, regionprops
import torch

from HiPas_postprocessing.connectivity import select_region
from HiPas_postprocessing.distance_transform import perform_distance_trans
from HiPas_postprocessing.get_skeleton import soft_dilate, soft_erode, soft_skel



def mm_to_vox(mm, spacing):
    """Convert millimeter threshold to voxel count (conservative: use min spacing)"""
    return max(1, int(round(mm / float(min(spacing)))))

def to_tensor5(x_np):
    """(D,H,W) -> [1,1,D,H,W] torch.Tensor(float32)"""
    return torch.from_numpy(x_np[np.newaxis, np.newaxis, ...].astype(np.float32))

def soft_closing(mask_np, steps=1):
    """Soft close operation: dilation and erosion are performed the same number of times"""
    x = to_tensor5(mask_np)
    for _ in range(steps):
        x = soft_dilate(x)
    for _ in range(steps):
        x = soft_erode(x)
    return (x.squeeze().numpy() > 0).astype(np.float32)

def filter_by_area_and_distance(mask_np, min_size_vox, near_mask_np, near_radius_vox):
    """
    Distance-aware filtering:
    Keep components if (area >= min_size) OR (min distance to near_mask <= near_radius)
    """
    lab = label((mask_np > 0).astype(np.uint8), connectivity=1)  
    if lab.max() == 0:
        return np.zeros_like(mask_np, dtype=np.float32)

    near = (near_mask_np > 0).astype(np.float32)
    dist = perform_distance_trans(near)

    out = np.zeros_like(mask_np, dtype=np.uint8)
    for r in regionprops(lab):
        comp = (lab == r.label)
        dmin = dist[comp].min() if r.area > 0 else 0
        if (r.area >= min_size_vox) or (dmin <= near_radius_vox):
            out[comp] = 1
    return out.astype(np.float32)


def constrained_bridge(mask_np, corridor_radius_vox=2, dilate_steps=2, final_erode=True):
    """
    Restricted bridging: dilate only within a “corridor” (<= radius from existing vessels)
    """
    base = (mask_np > 0).astype(np.float32)
    dist = perform_distance_trans(base)  # Voxel distance from background to foreground
    corridor = (dist <= corridor_radius_vox).astype(np.float32)

    x = to_tensor5(base)
    cor = to_tensor5(corridor)
    for _ in range(dilate_steps):
        x = soft_dilate(x)
        x = torch.minimum(x, cor)  # Restricted to the corridor
    if final_erode:
        x = soft_erode(x)        
    return (x.squeeze().numpy() > 0).astype(np.float32)



def resolve_conflicts(new_a, new_b, trachea, orig_a, orig_b):
    """
    Handling cases where arteries and veins overlap and intersect with the trachea.
    """
    a = (new_a > 0).astype(np.uint8)
    b = (new_b > 0).astype(np.uint8)
    tr = (trachea > 0).astype(np.uint8)

    a[tr == 1] = 0
    b[tr == 1] = 0

    overlap = (a == 1) & (b == 1)
    if overlap.any():
        da = perform_distance_trans((orig_a > 0).astype(np.float32))
        db = perform_distance_trans((orig_b > 0).astype(np.float32))

        choose_a = overlap & (da <= db)
        choose_b = overlap & (db < da)
        b[choose_a] = 0
        a[choose_b] = 0

    return a.astype(np.float32), b.astype(np.float32)

def count_skeleton_endpoints(mask_np, skel_iters=6, skel_thresh=0.1):
    """Calculate the number of skeleton endpoints"""
    if mask_np.max() == 0:
        return 0
    sk = soft_skel(mask_np.astype(np.float32), iter_=skel_iters)
    sk = (sk > skel_thresh).astype(np.uint8)
    kernel = np.ones((3,3,3), np.uint8); kernel[1,1,1] = 0
    deg = ndi.convolve(sk, kernel, mode='constant', cval=0)
    endpoints = ((sk == 1) & (deg == 1))
    return int(endpoints.sum())

def process_one_label(mask_np, spacing, args, name="artery"):
    """
    Complete process of vascular reconnection
    """
    # fill small gaps
    if args.close_steps > 0:
        mask_np = soft_closing(mask_np, steps=args.close_steps)

    # constructing the maximum connected domain, cleaning up distant points
    main = select_region(mask_np, num=1, thre=0)
    near_vox = mm_to_vox(args.near_main_mm, spacing)
    min_size_vox = int(args.min_size)
    mask_np = filter_by_area_and_distance(mask_np, min_size_vox, main, near_vox)

    # restricted bridge 
    corridor_vox = mm_to_vox(args.corridor_mm, spacing)
    if args.corridor_dilate > 0:
        bridged = constrained_bridge(mask_np,
                                     corridor_radius_vox=corridor_vox,
                                     dilate_steps=args.corridor_dilate,
                                     final_erode=True)
        base = (mask_np > 0).astype(np.float32)                

        added = ((bridged > 0) & (base == 0)).astype(np.float32)        
        thin  = (soft_skel(added, iter_=5) > 0.1).astype(np.float32)     
        near1 = (perform_distance_trans(base) <= 1).astype(np.float32)   
        mask_np = ((base > 0) | ((thin > 0) & (near1 > 0))).astype(np.float32)

    main = select_region(mask_np, num=1, thre=0)
    mask_np = filter_by_area_and_distance(mask_np, min_size_vox, main, near_vox)
    ep = count_skeleton_endpoints(mask_np, skel_iters=args.skel_iters, skel_thresh=0.1)

    stats = {
        f"{name}_endpoints": ep,
        f"{name}_voxels": int(mask_np.sum())
    }
    return mask_np, stats


def main():
    ap = argparse.ArgumentParser(description="Vessel reconnection post-processing")
    ap.add_argument("--in_path", required=True, help=".nii or .nii.gz input path")
    ap.add_argument("--out_path", required=True, help="output .nii.gz path")

    ap.add_argument("--min_size", type=int, default=150, help="Minimum connected component size to keep")
    ap.add_argument("--close_steps", type=int, default=1, help="Soft close operation steps")
    ap.add_argument("--near_main_mm", type=float, default=3.0, help="Judgement radius close to the main body")
    ap.add_argument("--corridor_mm", type=float, default=2.0, help="Corridor radius of restricted bridge")
    ap.add_argument("--corridor_dilate", type=int, default=2, help="Number of steps expanded in the corridor")
    ap.add_argument("--skel_iters", type=int, default=6, help="Number of iterations of the soft skeleton, used for endpoint statistics")
    ap.add_argument("--report_only", action="store_true", help="Only count the number of endpoints/voxels, do not write to file.")
    args = ap.parse_args()


    nii = nib.load(args.in_path)
    vol = nii.get_fdata().astype(np.int16)
    spacing = nii.header.get_zooms()  

    mask_tr = (vol == 1).astype(np.float32)
    mask_a0 = (vol == 2).astype(np.float32)  
    mask_v0 = (vol == 3).astype(np.float32)  

 
    a_proc, a_stats = process_one_label(mask_a0.copy(), spacing, args, name="artery")
    v_proc, v_stats = process_one_label(mask_v0.copy(), spacing, args, name="vein")


    a_final, v_final = resolve_conflicts(a_proc, v_proc, mask_tr, mask_a0, mask_v0)

    a_main = select_region(a_final, num=1, thre=0)
    v_main = select_region(v_final, num=1, thre=0)
    near_vox = mm_to_vox(args.near_main_mm, spacing)
    a_final = filter_by_area_and_distance(a_final, args.min_size, a_main, near_vox)
    v_final = filter_by_area_and_distance(v_final, args.min_size, v_main, near_vox)

    a_stats["artery_endpoints_final"] = count_skeleton_endpoints(a_final, skel_iters=args.skel_iters)
    v_stats["vein_endpoints_final"]   = count_skeleton_endpoints(v_final, skel_iters=args.skel_iters)
    a_stats["artery_voxels_final"]    = int(a_final.sum())
    v_stats["vein_voxels_final"]      = int(v_final.sum())

    print("[Stats]")
    for k, v in {**a_stats, **v_stats}.items():
        print(f"  {k}: {v}")

    if not args.report_only:
        out = np.zeros_like(vol, dtype=np.int16)
        out[mask_tr > 0] = 1
        out[a_final > 0] = 2
        out[v_final > 0] = 3
        nii_out = nib.Nifti1Image(out, nii.affine, nii.header)
        nib.save(nii_out, args.out_path)
        print(f"\nSaved -> {args.out_path}")

if __name__ == "__main__":
    main()