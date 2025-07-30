import torch
import math
import os
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset_1 import get_Dataloaders_new
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
import time
from evaluation import calculate_accuracy,calculate_dice,calculate_recall
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import BCEWithLogitsLoss



def save_probability_maps_from_output(output, num_classes=3, save_dir="prob_maps", prefix="epoch"):
    os.makedirs(save_dir, exist_ok=True)

    # Softmax 转成概率
    prob_map = torch.softmax(output, dim=1)  

    for class_index in range(num_classes):
        for slice_index in range(prob_map.shape[2]):  # 遍历深度 D
            prob_slice = prob_map[0, class_index, slice_index, :, :].detach().cpu().numpy()

            plt.imshow(prob_slice, cmap='jet')
            plt.colorbar(label='Probability')
            plt.title(f'Class {class_index} - Slice {slice_index}')

            save_path = os.path.join(save_dir, f"{prefix}_class{class_index}_slice{slice_index}.png")
            plt.savefig(save_path)
            plt.close()



if BACKGROUND_AS_CLASS:
  NUM_CLASSES += 1

# writer = SummaryWriter("runs")
model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES,cross_hair=True)
train_transforms = train_transform
val_transforms = val_transform

if torch.cuda.is_available() and TRAIN_CUDA:
  model = model.cuda()
  train_transforms = train_transform_cuda
  val_transforms = val_transform_cuda
  print("Use GPU")

train_dataloader, val_dataloader, _ = get_Dataloaders_new(train_transforms = train_transforms, val_transforms = val_transforms, test_transforms = val_transforms)



weights = torch.Tensor(BCE_WEIGHTS)
weights= weights.to("cuda")

# criterion = CrossEntropyLoss(weight=weights,ignore_index=-999)
# print("Use CrossEntropyLoss")
pos_w = torch.tensor(BCE_WEIGHTS).view(1, -1, 1, 1, 1).to("cuda")
criterion = BCEWithLogitsLoss(pos_weight=pos_w)
print("Use CEWithLogitsLoss")



optimizer = Adam(params=model.parameters(),lr=0.0001)

min_valid_loss = math.inf
epoch_times = []
accuracy_list = []
train_loss_list = []
valid_loss_list = []

for epoch in range(TRAINING_EPOCH):

    epoch_start = time.time()
    train_loss = 0.0
    model.train()
    for data in train_dataloader:
        image, ground_truth = data['image'], data['label']
        

        ground_truth = ground_truth.squeeze(1).long()
        # ground_truth[ground_truth < 0] = -999

        optimizer.zero_grad()
        target = model(image)

        y_true = F.one_hot(ground_truth, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        loss = criterion(target, y_true)
        # loss = criterion(target, ground_truth)
        # print(f'Train loss:{loss}')
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
	
    dice_sums = [0.0] * NUM_CLASSES
    recall_sums = [0.0] * NUM_CLASSES
    accuracy=0.0
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
      for data in val_dataloader:
        image, ground_truth = data['image'], data['label']
        

        ground_truth = ground_truth.squeeze(1)
        ground_truth = ground_truth.long()

        target = model(image)
        y_true = F.one_hot(ground_truth, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        loss = criterion(target, y_true)
        # loss = criterion(target,ground_truth)
        # print(f'Valid loss:{loss}')
        valid_loss += loss.item()
        accuracy += calculate_accuracy(target, ground_truth)
        dice_scores = calculate_dice(target, ground_truth, NUM_CLASSES)
        for i, dice in enumerate(dice_scores):
            dice_sums[i] += dice
        recalls = calculate_recall(target, ground_truth, NUM_CLASSES)
        for i, r in enumerate(recalls):
            recall_sums[i] += r


    # writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    # writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)

    epoch_end = time.time()
    avg_train=train_loss / len(train_dataloader)
    avg_valid=valid_loss / len(val_dataloader)
    train_loss_list.append(avg_train)
    valid_loss_list.append(avg_valid)
   
    avg_accuracy = accuracy / len(val_dataloader)
    accuracy_list.append(avg_accuracy)
    print(f'Validation Accuracy: {avg_accuracy * 100:.2f}%')

    avg_dice_per_class = [
      dice_sums[i] / len(val_dataloader)
      for i in range(NUM_CLASSES)
    ]
    for i, d in enumerate(avg_dice_per_class):
      print(f"Dice Class {i}: {d:.4f}")

    avg_recall_per_class = [
      recall_sums[i] / len(val_dataloader)
      for i in range(NUM_CLASSES)
    ]
    for i, r in enumerate(avg_recall_per_class):
      print(f"Recall Class {i}: {r:.4f}")



    print(f'Epoch {epoch+1} \t\t Training Loss: {avg_train} \t\t Validation Loss: {avg_valid}')
    epoch_times.append(epoch_end - epoch_start)
    


    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{avg_valid:.6f}) \t Saving The Model')
        min_valid_loss = avg_valid
        # Saving State Dict
        # torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, "checkpoint.pth")
    
save_probability_maps_from_output(
    target, 
    num_classes=NUM_CLASSES, 
    save_dir="prob_maps", 
    prefix=f"epoch_{TRAINING_EPOCH}"
)


print("\n===== Average Accuracy Every 10 Epochs =====")
for i in range(0, len(accuracy_list), 10):
    chunk = accuracy_list[i:i+10]
    avg_acc = sum(chunk) / len(chunk)
    print(f"Epoch {i+1} to {i+len(chunk)} average accuracy: {avg_acc * 100:.2f}%")

print("\n===== Epoch Time Summary =====")
for i in range(0, len(epoch_times), 10):
    chunk = epoch_times[i:i+10]
    avg_chunk = sum(chunk) / len(chunk)
    print(f"Epoch {i+1} to {i+len(chunk)} average time: {avg_chunk:.2f} seconds")

total_avg = sum(epoch_times) / len(epoch_times)
print(f"\nOverall average time for all {TRAINING_EPOCH} epochs: {total_avg:.2f} seconds")

plt.figure()
plt.plot(range(1, TRAINING_EPOCH+1), train_loss_list, label='Train Loss')
plt.plot(range(1, TRAINING_EPOCH+1), valid_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.savefig('loss_epoch.png')
plt.show()

