import monai
import torch
import os
import time
import numpy as np
from unet3d import UNet3D
from dataset import get_Dataloaders_new
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
from config import NUM_CLASSES, IN_CHANNELS, BACKGROUND_AS_CLASS, TRAIN_CUDA
from evaluation import calculate_accuracy, calculate_dice, calculate_recall,per_class_accuracy,calculate_iou,calculate_hausdorff


if BACKGROUND_AS_CLASS:
    NUM_CLASSES += 1


train_transforms = train_transform
val_transforms = val_transform

if torch.cuda.is_available() and TRAIN_CUDA:
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    device = torch.device("cuda")
    print("Use GPU")
else:
    device = torch.device("cpu")


model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
model.to(device)



torch.serialization.add_safe_globals([
    monai.utils.enums.MetaKeys,
    monai.utils.enums.SpaceKeys,
    monai.utils.enums.TraceKeys
])
checkpoint = torch.load("checkpoint_noch.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print("Test no cross_hair model")
model.eval()


_, _, test_dataloader = get_Dataloaders_new(train_transforms = train_transforms, val_transforms = val_transforms, test_transforms = val_transforms)

os.makedirs("predictions_noch", exist_ok=True)
prediction_times = []
accuracy = 0.0
dice_sums = [0.0] * NUM_CLASSES
recall_sums = [0.0] * NUM_CLASSES
accuracy_sums = [0.0] * NUM_CLASSES
iou_sums = [0.0] * NUM_CLASSES
hd_sums = [0.0] * NUM_CLASSES

with torch.no_grad():
    for idx, data in enumerate(test_dataloader):
        image, ground_truth = data['image'], data['label']
        ground_truth = ground_truth.squeeze(1).long()

        start_time = time.time()
        output = model(image)
        end_time = time.time()

        elapsed = end_time - start_time
        prediction_times.append(elapsed)
        print(f"Sample {idx+1} prediction time: {elapsed:.4f} seconds")

        np.save(f"predictions_noch/output_{idx+1}.npy", output.cpu().numpy())
    
        acc = calculate_accuracy(output, ground_truth)
        dice_scores = calculate_dice(output, ground_truth, NUM_CLASSES)
        recalls = calculate_recall(output, ground_truth, NUM_CLASSES)
        per_class_acc = per_class_accuracy(output, ground_truth, NUM_CLASSES)
        iou_scores = calculate_iou(output, ground_truth, NUM_CLASSES)
        hd_scores = calculate_hausdorff(output, ground_truth, NUM_CLASSES)

        print(f"\n=== Sample {idx+1} ===")
        print(f"Prediction time: {elapsed:.4f} seconds")
        print(f"Accuracy: {acc:.4f}")
        for i in range(NUM_CLASSES):
            print(f"Class {i} -> Dice: {dice_scores[i]:.4f}, "
                  f"Recall: {recalls[i]:.4f}, "
                  f"Accuracy: {per_class_acc[i]:.4f}, "
                  f"IoU: {iou_scores[i]:.4f}, "
                  f"Hausdorff: {hd_scores[i]:.4f}")
            

        accuracy += acc
        for i in range(NUM_CLASSES):
            dice_sums[i] += dice_scores[i]
            recall_sums[i] += recalls[i]
            accuracy_sums[i] += per_class_acc[i]
            iou_sums[i] += iou_scores[i]
            if not np.isnan(hd_scores[i]):
                hd_sums[i] += hd_scores[i]


avg_accuracy = accuracy / len(test_dataloader)
print(f"Test Accuracy: {avg_accuracy * 100:.2f}%")


avg_dice_per_class = [dice_sums[i] / len(test_dataloader) for i in range(NUM_CLASSES)]
avg_recall_per_class = [recall_sums[i] / len(test_dataloader) for i in range(NUM_CLASSES)]
avg_accuracy_per_class = [accuracy_sums[i] / len(test_dataloader) for i in range(NUM_CLASSES)]
avg_iou_per_class = [iou_sums[i] / len(test_dataloader) for i in range(NUM_CLASSES)]
avg_hd_per_class = [hd_sums[i] / len(test_dataloader) for i in range(NUM_CLASSES)]

for i, d in enumerate(avg_dice_per_class):
    print(f"Dice Class {i}: {d:.4f}")

for i, r in enumerate(avg_recall_per_class):
    print(f"Recall Class {i}: {r:.4f}")

for i, r in enumerate(avg_accuracy_per_class):
      print(f"Accuracy Class {i}: {r:.4f}")

for i, iou in enumerate(avg_iou_per_class):
    print(f"IoU Class {i}: {iou:.4f}")

for i, h in enumerate(avg_hd_per_class):
    print(f"Hausdorff Distance Class {i}: {h:.4f}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

avg_pred_time = sum(prediction_times) / len(prediction_times)
print(f"\nAverage prediction time: {avg_pred_time:.4f} seconds")

