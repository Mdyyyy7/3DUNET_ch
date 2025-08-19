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

import sys
sys.path.append(r"HiPas_postprocessing")
from connectivity import select_region
from distance_transform import perform_distance_trans
from get_skeleton import soft_dilate, soft_erode, soft_skel

# -----------------------
# 工具函数
# -----------------------
def mm_to_vox(mm, spacing):
    """把毫米阈值换算到体素数。这里用最小 spacing 做保守估计。"""
    return max(1, int(round(mm / float(min(spacing)))))

def to_tensor5(x_np):
    """(D,H,W) -> [1,1,D,H,W] torch.Tensor(float32)"""
    return torch.from_numpy(x_np[np.newaxis, np.newaxis, ...].astype(np.float32))

def soft_closing(mask_np, steps=1):
    """软闭运算：膨胀 steps 次，再腐蚀 steps 次。输入/输出为 {0,1} float32。"""
    x = to_tensor5(mask_np)
    for _ in range(steps):
        x = soft_dilate(x)
    for _ in range(steps):
        x = soft_erode(x)
    return (x.squeeze().numpy() > 0).astype(np.float32)

def filter_by_area_and_distance(mask_np, min_size_vox, near_mask_np, near_radius_vox):
    """
    距离感知清理：保留 (面积>=min_size) 或 (到 near_mask 的最近距离 <= near_radius)
    其余删除。
    near_mask 通常是主体（最大连通域）。
    """
    lab = label((mask_np > 0).astype(np.uint8), connectivity=1)  # 3D: 6-邻域
    if lab.max() == 0:
        return np.zeros_like(mask_np, dtype=np.float32)

    # 到 near_mask 的距离（使用你提供的 perform_distance_trans：背景到1的距离）
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
    受限补桥：仅在“距离现有血管 <= corridor_radius_vox”的走廊里做膨胀，避免乱连。
    """
    base = (mask_np > 0).astype(np.float32)
    dist = perform_distance_trans(base)  # 背景->前景的体素距离
    corridor = (dist <= corridor_radius_vox).astype(np.float32)

    x = to_tensor5(base)
    cor = to_tensor5(corridor)
    for _ in range(dilate_steps):
        x = soft_dilate(x)
        x = torch.minimum(x, cor)  # 限制在走廊里
    if final_erode:
        x = soft_erode(x)          # 收回粗边
    return (x.squeeze().numpy() > 0).astype(np.float32)



def resolve_conflicts(new_a, new_b, trachea, orig_a, orig_b):
    """
    处理动静脉互相重叠 & 与气管相交：
    - 与气管相交处，一律清 0。
    - 动静脉重叠处：根据到各自“原始掩膜”的距离，分配给更近的一类。
    """
    a = (new_a > 0).astype(np.uint8)
    b = (new_b > 0).astype(np.uint8)
    tr = (trachea > 0).astype(np.uint8)

    # 先去掉与气管相交处
    a[tr == 1] = 0
    b[tr == 1] = 0

    overlap = (a == 1) & (b == 1)
    if overlap.any():
        # 到原动脉/静脉掩膜的距离（背景->原掩膜）
        da = perform_distance_trans((orig_a > 0).astype(np.float32))
        db = perform_distance_trans((orig_b > 0).astype(np.float32))
        # 在重叠处，根据更近的原类别来决定归属
        choose_a = overlap & (da <= db)
        choose_b = overlap & (db < da)
        b[choose_a] = 0
        a[choose_b] = 0

    return a.astype(np.float32), b.astype(np.float32)

def count_skeleton_endpoints(mask_np, skel_iters=6, skel_thresh=0.1):
    """骨架端点计数（26 邻域度为1的骨架点）。"""
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
    针对一个标签（2 或 3）的完整重连流程。
    返回重连后的 {0,1} float32 掩膜，以及若干统计。
    """
    # Step 1: 小闭运算补短缺口（先补后筛）
    if args.close_steps > 0:
        mask_np = soft_closing(mask_np, steps=args.close_steps)

    # Step 2: 构建主体（最大连通域），并执行“距离感知清理”
    main = select_region(mask_np, num=1, thre=0)
    near_vox = mm_to_vox(args.near_main_mm, spacing)
    min_size_vox = int(args.min_size)
    mask_np = filter_by_area_and_distance(mask_np, min_size_vox, main, near_vox)

    # Step 3: 受限补桥（只在走廊内膨胀）
    corridor_vox = mm_to_vox(args.corridor_mm, spacing)
    if args.corridor_dilate > 0:
        bridged = constrained_bridge(mask_np,
                                     corridor_radius_vox=corridor_vox,
                                     dilate_steps=args.corridor_dilate,
                                     final_erode=True)
        # 合并（或直接替换）——更稳的是并回原掩膜
        base = (mask_np > 0).astype(np.float32)                 # 原掩膜

        added = ((bridged > 0) & (base == 0)).astype(np.float32)         # 仅新增部分
        thin  = (soft_skel(added, iter_=5) > 0.1).astype(np.float32)     # 细桥化为单体素
        near1 = (perform_distance_trans(base) <= 1).astype(np.float32)   # 仅保留离原掩膜≤1体素

        mask_np = ((base > 0) | ((thin > 0) & (near1 > 0))).astype(np.float32)

    # Step 4: 再次距离感知清理（避免补桥后产生远端连通）
    main = select_region(mask_np, num=1, thre=0)
    mask_np = filter_by_area_and_distance(mask_np, min_size_vox, main, near_vox)

    # Step 5: 统计骨架端点
    ep = count_skeleton_endpoints(mask_np, skel_iters=args.skel_iters, skel_thresh=0.1)

    stats = {
        f"{name}_endpoints": ep,
        f"{name}_voxels": int(mask_np.sum())
    }
    return mask_np, stats

# -----------------------
# 主流程
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Vessel reconnection post-processing for labels 2 (artery) and 3 (vein).")
    ap.add_argument("--in_path", required=True, help=".nii or .nii.gz input path")
    ap.add_argument("--out_path", required=True, help="output .nii.gz path")
    # 参数（可按数据微调）
    ap.add_argument("--min_size", type=int, default=150, help="保留的最小连通域体素数（面积阈值）")
    ap.add_argument("--close_steps", type=int, default=1, help="软闭运算步数（补短缺口，0 表示不做）")
    ap.add_argument("--near_main_mm", type=float, default=3.0, help="距离主体近的判定半径（mm），近小段会保留")
    ap.add_argument("--corridor_mm", type=float, default=2.0, help="受限补桥的走廊半径（mm）")
    ap.add_argument("--corridor_dilate", type=int, default=2, help="走廊内膨胀的步数（越大越容易连上，也更易误连）")
    ap.add_argument("--skel_iters", type=int, default=6, help="软骨架迭代次数，用于端点统计")
    ap.add_argument("--report_only", action="store_true", help="只统计端点/体素数，不写文件")
    args = ap.parse_args()

    # 读取
    nii = nib.load(args.in_path)
    vol = nii.get_fdata().astype(np.int16)
    spacing = nii.header.get_zooms()  # (sx, sy, sz) in mm

    mask_tr = (vol == 1).astype(np.float32)
    mask_a0 = (vol == 2).astype(np.float32)  # 原始动脉
    mask_v0 = (vol == 3).astype(np.float32)  # 原始静脉

    # 分别处理
    a_proc, a_stats = process_one_label(mask_a0.copy(), spacing, args, name="artery")
    v_proc, v_stats = process_one_label(mask_v0.copy(), spacing, args, name="vein")

    # 冲突处理：与气管/对方的交集清理，并用“到原掩膜的距离”裁决重叠归属
    a_final, v_final = resolve_conflicts(a_proc, v_proc, mask_tr, mask_a0, mask_v0)

    # 最终再做一次“距离主体近”的约束——忽略远离主体的残留端点/碎段
    a_main = select_region(a_final, num=1, thre=0)
    v_main = select_region(v_final, num=1, thre=0)
    near_vox = mm_to_vox(args.near_main_mm, spacing)
    a_final = filter_by_area_and_distance(a_final, args.min_size, a_main, near_vox)
    v_final = filter_by_area_and_distance(v_final, args.min_size, v_main, near_vox)

    # 终局统计
    a_stats["artery_endpoints_final"] = count_skeleton_endpoints(a_final, skel_iters=args.skel_iters)
    v_stats["vein_endpoints_final"]   = count_skeleton_endpoints(v_final, skel_iters=args.skel_iters)
    a_stats["artery_voxels_final"]    = int(a_final.sum())
    v_stats["vein_voxels_final"]      = int(v_final.sum())

    # 结果输出
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