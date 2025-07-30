import monai
import torch
import torch.nn.functional as F
from unet3d import UNet3D
from dataset_1 import get_Dataloaders_new
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
from config import NUM_CLASSES, IN_CHANNELS, BACKGROUND_AS_CLASS, TRAIN_CUDA
from evaluation import calculate_accuracy, calculate_dice, calculate_recall


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


model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, cross_hair=True)
model.to(device)

torch.serialization.add_safe_globals([
    monai.utils.enums.MetaKeys,
    monai.utils.enums.SpaceKeys,
    monai.utils.enums.TraceKeys
])
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


_, _, test_dataloader = get_Dataloaders_new(train_transforms = train_transforms, val_transforms = val_transforms, test_transforms = val_transforms)


accuracy = 0.0
dice_sums = [0.0] * NUM_CLASSES
recall_sums = [0.0] * NUM_CLASSES

with torch.no_grad():
    for data in test_dataloader:
        image, ground_truth = data['image'], data['label']
        ground_truth = ground_truth.squeeze(1).long()

        output = model(image)

        accuracy += calculate_accuracy(output, ground_truth)
        dice_scores = calculate_dice(output, ground_truth, NUM_CLASSES)
        for i, d in enumerate(dice_scores):
            dice_sums[i] += d
        recalls = calculate_recall(output, ground_truth, NUM_CLASSES)
        for i, r in enumerate(recalls):
            recall_sums[i] += r


avg_accuracy = accuracy / len(test_dataloader)
print(f"Test Accuracy: {avg_accuracy * 100:.2f}%")


avg_dice_per_class = [dice_sums[i] / len(test_dataloader) for i in range(NUM_CLASSES)]
avg_recall_per_class = [recall_sums[i] / len(test_dataloader) for i in range(NUM_CLASSES)]

for i, d in enumerate(avg_dice_per_class):
    print(f"Dice Class {i}: {d:.4f}")

for i, r in enumerate(avg_recall_per_class):
    print(f"Recall Class {i}: {r:.4f}")
