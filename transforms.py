from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureType,
    MapTransform,
    Resized,
    DivisiblePadd,
    Lambdad,
    GaussianSmoothd
)
import numpy as np
from typing import Union, List, Dict
import cv2



class CLAHETransform3D(MapTransform):
    def __init__(
        self,
        keys: Union[str, List[str]],
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8)
    ):
        super().__init__(keys)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data: Dict):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # 获取图像数据，形状应为 [C, D, H, W]

            if img.ndim != 4:
                raise ValueError(f"期望图像形状为 4D [C, D, H, W]，但得到的是 {img.shape}")

            c, d_len, h, w = img.shape
            enhanced = np.zeros_like(img)

            # 遍历每个通道和切片
            for c_idx in range(c):
                for d_idx in range(d_len):
                    slice_2d = img[c_idx, d_idx].astype(np.uint8)  # 取出单个 [H, W] 切片
                    clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                    enhanced_slice = clahe.apply(slice_2d)
                    enhanced[c_idx, d_idx] = enhanced_slice

            d[key] = enhanced
        return d





# 训练数据转化过程
train_transform = Compose(
[
  EnsureType(data_type='tensor'),
  # 添加维度
  #EnsureChannelFirstd(keys=["image", "label"]),
  # 统一体素尺寸
  Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
  # 随机翻转
  RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
  RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
  RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),

  # 增强对比度
  # transforms.ColorJitter(contrast=0.5),
  # 高斯滤波器
  # GaussianSmoothd(keys=["image"], sigma_x=1.0, sigma_y=1.0, sigma_z=1.0)
  # CLAHE
  # CLAHETransform3D(keys=["image"], clip_limit=2.0, tile_grid_size=(8, 8)),

  # 归一化
  NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),

  # 强度缩放和偏移
  RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
  RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
  # 补全16倍数
  DivisiblePadd(k=16, keys=["image", "label"]),

  Resized(keys=["image", "label"], spatial_size=(96, 96, 96)),

  ToTensord(keys=['image', 'label'])
]
)



# Cuda版本
train_transform_cuda = Compose(
[
  EnsureType(data_type='tensor'),
  # #EnsureChannelFirstd(keys=["image", "label"]),
  # Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
  # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
  # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
  # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),

  # # 高斯滤波器
  # # GaussianSmoothd(keys=["image"], sigma=[1.0, 1.0, 1.0]),
  # # CLAHE
  # # CLAHETransform3D(keys=["image"], clip_limit=2.0, tile_grid_size=(8, 8)),

  # NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
  # RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
  # RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
  
  Lambdad(
    keys="image",
    func=lambda x: (x.clamp(min=-1000, max=400) + 1000) / 1400
  ),
  DivisiblePadd(k=16, keys=["image", "label"]),

  # Resized(keys=["image", "label"], spatial_size=(48,48,48), mode=("trilinear", "nearest")),

  ToTensord(keys=['image', 'label'], device='cuda')
]
)


# 测试数据和验证数据
val_transform = Compose(
[
  EnsureType(data_type='tensor'),
  #EnsureChannelFirstd(keys=["image", "label"]),

  NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
  DivisiblePadd(k=16, keys=["image", "label"]),

  Resized(keys=["image", "label"], spatial_size=(96, 96, 96)),

  ToTensord(keys=['image', 'label'])
]
)


# cuda版本
val_transform_cuda = Compose(
[
  EnsureType(data_type='tensor'),
  #EnsureChannelFirstd(keys=["image", "label"]),


  # NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
  
  Lambdad(
    keys="image",
    func=lambda x: (x.clamp(min=-1000, max=400) + 1000) / 1400
  ),
  DivisiblePadd(k=16, keys=["image", "label"]),

  # Resized(keys=["image", "label"], spatial_size=(48,48,48),  mode=("trilinear", "nearest")),

  ToTensord(keys=['image', 'label'], device='cuda')
]
)