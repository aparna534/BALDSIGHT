# config.py

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

NUM_CLASSES = 5
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DATA_DIR = "./datasets/norwood_images"
MODEL_SAVE_DIR = "./saved_models"
SEED = 42

