import torch

NUM_EPOCHS_INIT = 10
NUM_EPOCHS = 40    # 2000 in paper implementation
BATCH_SIZE = 64
LEARNING_RATE = 0.05
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_IMG_DIR = '...'
TEST_IMG_DIR = '...'

CROP_SIZE = 96
SCALE_FACTOR = 4

SAVE_CHECKPOINT_EPOCH = 10