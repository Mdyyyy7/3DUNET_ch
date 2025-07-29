import nibabel as nib
import tarfile
import numpy as np
import os
import json
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import copy
import torch
from config import (
    DATASET_PATH, TRAIN_VAL_TEST_SPLIT,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)


# 解压文件
def DecompressFile(dir):
  try:
      print("Extracting tar file ...")
      tarfile.open(dir).extractall('./Datasets')
  except FileNotFoundError:
      raise FileNotFoundError(f"Dataset file not found at {dir}. Please check the DATASET_PATH.")
  except Exception as e:
      raise RuntimeError(f"Failed to decompress file {dir}: {e}")
  print("Decompression completed!")
  return



class LungVesselSegmentation(Dataset):
  def __init__(self, path, split_ratios=[0.6, 0.2, 0.2], transforms=None, mode=None):
    self.image_dir = os.path.join(path, "image")
    self.label_dir = os.path.join(path, "label")
    self.transforms = transforms
    self.mode = mode

    self.file_list = [f for f in os.listdir(self.image_dir) if f.endswith(".mhd")]
    # self.file_list.sort()
    # shuffle(self.file_list)

    # 划分数据集
    total = len(self.file_list)
    num_list = [int(r * total) for r in split_ratios]
    if sum(num_list) < total:
        num_list[0] += total - sum(num_list)

    self.train = self.file_list[:num_list[0]]
    self.val = self.file_list[num_list[0]:num_list[0]+num_list[1]]
    self.test = self.file_list[num_list[0]+num_list[1]:]

# 设置训练，验证或测试模式
  def set_mode(self, mode):
        self.mode = mode

  def __len__(self):
    if self.mode == "train":
        return len(self.train)
    elif self.mode == "val":
        return len(self.val)
    elif self.mode == "test":
        return len(self.test)
    return len(self.train)

  def __getitem__(self, idx):
    # 获取文件名
    if self.mode == "train":
        file_name = self.train[idx].replace(".mhd", "")
    elif self.mode == "val":
        file_name = self.val[idx].replace(".mhd", "")
    elif self.mode == "test":
        file_name = self.test[idx].replace(".mhd", "")
    else:
        file_name = self.file_list[idx].replace(".mhd", "")

    image_path = os.path.join(self.image_dir, file_name + ".mhd")
    label_base = file_name.rsplit(".7", 1)[0]
    label_path = os.path.join(self.label_dir, label_base + "_fullAnnotations.mhd")

    # 获取三维体素数据
    img_item = sitk.ReadImage(image_path)
    label_item = sitk.ReadImage(label_path)

    img = sitk.GetArrayFromImage(img_item)
    label = sitk.GetArrayFromImage(label_item)

    img = np.moveaxis(img, -1, 0)
    label = np.moveaxis(label, -1, 0)
    img = np.expand_dims(img, axis=0)
    label = np.expand_dims(label, axis=0)

    # print(f"images.shape1: {img.shape}")
    # unique_labels, counts = np.unique(label, return_counts=True)
    # print(f"labels.shape1: {unique_labels}")

    # 数据处理
    processed_out = {'name': file_name, 'image': img, 'label': label}
    if self.transforms:
      if self.mode == "train":
          processed_out = self.transforms[0](processed_out)
      elif self.mode == "val":
          processed_out = self.transforms[1](processed_out)
      elif self.mode == "test":
          processed_out = self.transforms[2](processed_out)
      else:
        processed_out = self.transforms(processed_out)

    # image = processed_out['image']
    # label = processed_out['label']

    # print(f"Image shape2: {image.shape}")
    # print(f"Label shape2: {label.shape}")

    return processed_out


def get_Dataloaders(train_transforms, val_transforms, test_transforms):
  dataset = LungVesselSegmentation(path=DATASET_PATH, split_ratios=TRAIN_VAL_TEST_SPLIT, transforms=[train_transforms, val_transforms, test_transforms])

  train_set, val_set, test_set = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
  train_set.set_mode('train')
  val_set.set_mode('val')
  test_set.set_mode('test')
  train_dataloader = DataLoader(dataset= train_set, batch_size= TRAIN_BATCH_SIZE, shuffle= False)
  val_dataloader = DataLoader(dataset= val_set, batch_size= VAL_BATCH_SIZE, shuffle= False)
  test_dataloader = DataLoader(dataset= test_set, batch_size= TEST_BATCH_SIZE, shuffle= False)

  return train_dataloader, val_dataloader, test_dataloader