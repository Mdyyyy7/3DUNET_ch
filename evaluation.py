import torch
from scipy.spatial.distance import directed_hausdorff
import numpy as np



def calculate_accuracy(prediction, ground_truth, ignore_index=-999):

    with torch.no_grad():
        _, predicted = torch.max(prediction, 1)
        mask = (ground_truth != ignore_index)
        correct = (predicted[mask] == ground_truth[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0
        return accuracy



def calculate_dice(prediction, ground_truth, num_classes, ignore_index=-999):

    with torch.no_grad():
        _, predicted = torch.max(prediction, 1)  
        dice_scores = []

        for cls in range(num_classes):
            pred_cls = (predicted == cls)
            gt_cls = (ground_truth == cls)
            mask = (ground_truth != ignore_index)

            pred_cls = pred_cls & mask
            gt_cls = gt_cls & mask

            intersection = (pred_cls & gt_cls).sum().item()
            pred_sum = pred_cls.sum().item()
            gt_sum = gt_cls.sum().item()

            if pred_sum + gt_sum == 0:
                dice = 1.0  
            else:
                dice = (2.0 * intersection) / (pred_sum + gt_sum)

            dice_scores.append(dice)
            #print(f"[Class {cls}] pred_sum: {pred_cls.sum().item()}, gt_sum: {gt_cls.sum().item()}, intersection: {intersection}")

        return dice_scores
    


def calculate_recall(prediction, ground_truth, num_classes, ignore_index=-999):
    """
     Calculate the recall rate for each category Recall
     Return: list of recall per class
    """
    with torch.no_grad():
        _, predicted = torch.max(prediction, 1) 
        recall_scores = []

        for cls in range(num_classes):
            pred_cls = (predicted == cls)
            gt_cls = (ground_truth == cls)
            mask = (ground_truth != ignore_index)

            pred_cls = pred_cls & mask
            gt_cls = gt_cls & mask

            TP = (pred_cls & gt_cls).sum().item()
            FN = (gt_cls & (~pred_cls)).sum().item()

            recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
            recall_scores.append(recall)

        return recall_scores

def per_class_accuracy(prediction, ground_truth, num_classes, ignore_index=None):
    with torch.no_grad():
        _, predicted = torch.max(prediction, 1)

        if ignore_index is not None:
            mask = (ground_truth != ignore_index)
            predicted = predicted[mask]
            ground_truth = ground_truth[mask]

        precisions = []
        for cls in range(num_classes):
            pred_cls_mask = (predicted == cls)
            total_pred_cls = pred_cls_mask.sum().item()

            if total_pred_cls == 0:
                precisions.append(0)
                continue

            correct_cls = ((predicted == cls) & (ground_truth == cls)).sum().item()
            precision_cls = correct_cls / total_pred_cls
            precisions.append(precision_cls)

    return precisions

def calculate_iou(prediction, ground_truth, num_classes, ignore_index=None):
    with torch.no_grad():
        _, predicted = torch.max(prediction, 1)

        if ignore_index is not None:
            mask = (ground_truth != ignore_index)
            predicted = predicted[mask]
            ground_truth = ground_truth[mask]
        iou_list=[]
        for cls in range(num_classes):
            pred_cls_mask = (predicted == cls)
            gt_cls_mask = (ground_truth == cls)

            intersection = (pred_cls_mask & gt_cls_mask).sum().float()
            union = (pred_cls_mask | gt_cls_mask).sum().float()

            if union == 0:
                iou = 1.0  
            else:
                iou = intersection / union
            iou_list.append(iou.item())

    return iou_list
            
def calculate_hausdorff(prediction, ground_truth, num_classes, ignore_index=-999):

    with torch.no_grad():
        _, predicted = torch.max(prediction, 1) 
        hd_scores = []

        for cls in range(num_classes):
            pred_cls = (predicted == cls)
            gt_cls = (ground_truth == cls)
            mask = (ground_truth != ignore_index)

            pred_cls = pred_cls & mask
            gt_cls = gt_cls & mask

            pred_pts = np.argwhere(pred_cls.cpu().numpy())
            gt_pts = np.argwhere(gt_cls.cpu().numpy())

            if len(pred_pts) == 0 or len(gt_pts) == 0:
                hd_scores.append(float('nan'))  
            else:
                hd1 = directed_hausdorff(pred_pts, gt_pts)[0]
                hd2 = directed_hausdorff(gt_pts, pred_pts)[0]
                hd_scores.append(max(hd1, hd2))

        return hd_scores