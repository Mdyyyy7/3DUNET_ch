from monai.transforms import (
    Compose,
    ToTensord,
    EnsureType,
    Resized,
    DivisiblePadd,
    Lambdad,
)



# Preprocessing of training data
train_transform = Compose(
[
  EnsureType(data_type='tensor'),

  # Normalize CT values
  Lambdad(
    keys="image",
    func=lambda x: (x.clamp(min=-1000, max=400) + 1000) / 1400
  ),
  DivisiblePadd(k=16, keys=["image", "label"]),
  # Change the image size
  Resized(keys=["image", "label"], spatial_size=(160,160,160), mode=("trilinear", "nearest")),
  ToTensord(keys=['image', 'label'])
]
)



# Cuda version of preprocessing for training data
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


# Preprocessing of test and validation data
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


# Cuda version of preprocessing for testing and validation data
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