"""
Dataset configurations:
    DATASET_PATH_NEW:  dataset path
    IN_CHANNELS: number of input channels
    NUM_CLASSES: number of output channels (number of categories)
    BACKGROUND_AS_CLASS: if true, set the background to a class
"""
DATASET_PATH_NEW="data_hipas"
IN_CHANNELS = 1
NUM_CLASSES = 2
BACKGROUND_AS_CLASS = True



"""
Training configuration:
    TRAIN_VAL_TEST_SPLI: the ratio of training set, validation set, and test set
    SPLIT_SEED: random seed, used to partition the dataset
    TRAINING_EPOCH: number of training epochs
    TRAIN_BATCH_SIZE: the batch size of the training DataLoader
    VAL_BATCH_SIZE: the batch size of the valid DataLoader
    TEST_BATCH_SIZE: the batch size of the test DataLoader
    TRAIN_CUDA: if true, training and inference will be performed on the GPU.
    BCE_WEIGHTS: weights for different classes in the binary cross-entropy loss
"""
TRAIN_VAL_TEST_SPLIT = [0.6, 0.2, 0.2]
SPLIT_SEED = 42
TRAINING_EPOCH = 130
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
TRAIN_CUDA = True
BCE_WEIGHTS = [0.004, 0.498,0.498]