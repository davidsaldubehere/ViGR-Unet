import torch
import os
BASE = "/scratch/das6859/UNetVGNNCAMUS/"
DATASET_PATH = os.path.join(BASE, "CAMUS ES Dataset")
BASE_OUTPUT = os.path.join(BASE, "output")

DATASET_PATH_TEST = os.path.join(DATASET_PATH, "test")
# base path of the dataset (changing to 'full' instead of 'train' for kfold)
DATASET_PATH = os.path.join(DATASET_PATH, "full")
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
IMAGE_DATASET_PATH_TEST = os.path.join(DATASET_PATH_TEST, "images")
MASK_DATASET_PATH_TEST = os.path.join(DATASET_PATH_TEST, "masks")

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_camus.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
