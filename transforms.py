from monai.transforms import (
    Compose,
    ToTensord,
    EnsureType,
    Resized,
    DivisiblePadd,
    Lambdad,
)







# 训练数据转化过程
train_transform = Compose(
[
  EnsureType(data_type='tensor'),

  Lambdad(
    keys="image",
    func=lambda x: (x.clamp(min=-1000, max=400) + 1000) / 1400
  ),
  DivisiblePadd(k=16, keys=["image", "label"]),
  Resized(keys=["image", "label"], spatial_size=(160,160,160), mode=("trilinear", "nearest")),
  ToTensord(keys=['image', 'label'])
]
)



# Cuda版本
train_transform_cuda = Compose(
[
  EnsureType(data_type='tensor'),

  Lambdad(
    keys="image",
    func=lambda x: (x.clamp(min=-1000, max=400) + 1000) / 1400
  ),
  DivisiblePadd(k=16, keys=["image", "label"]),
  Resized(keys=["image", "label"], spatial_size=(160,160,160), mode=("trilinear", "nearest")),
  ToTensord(keys=['image', 'label'], device='cuda')
]
)


# 测试数据和验证数据
val_transform = Compose(
[
  EnsureType(data_type='tensor'),

  Lambdad(
    keys="image",
    func=lambda x: (x.clamp(min=-1000, max=400) + 1000) / 1400
  ),
  DivisiblePadd(k=16, keys=["image", "label"]),
  Resized(keys=["image", "label"], spatial_size=(160,160,160),  mode=("trilinear", "nearest")),
  ToTensord(keys=['image', 'label'])
]
)


# cuda版本
val_transform_cuda = Compose(
[
  EnsureType(data_type='tensor'),

  Lambdad(
    keys="image",
    func=lambda x: (x.clamp(min=-1000, max=400) + 1000) / 1400
  ),
  DivisiblePadd(k=16, keys=["image", "label"]),
  Resized(keys=["image", "label"], spatial_size=(160,160,160),  mode=("trilinear", "nearest")),
  ToTensord(keys=['image', 'label'], device='cuda')
]
)