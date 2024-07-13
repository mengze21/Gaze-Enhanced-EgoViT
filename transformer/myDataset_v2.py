# ----------------------------------------
# this script is update from myDataset.py
# used for gaze_v2
# ----------------------------------------


import os
import glob
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class PreprocessedImageGazeDataset11(Dataset):
    """Load preprocessed image, gaze-hand-object features and label data."""
    def __init__(self, data_folder, gaze_folder, transform=None):
        self.data_folder = data_folder
        self.gaze_folder = gaze_folder
        self.transform = transform
        self.data_files = sorted(glob.glob(os.path.join(data_folder, '*.npz')))
        self.gaze_files = sorted(glob.glob(os.path.join(gaze_folder, '*.npy')))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        images = torch.tensor(data['images'], dtype=torch.float32)
        images = images / 255.0
        if self.transform:
            images = self.transform(images)
        images = images.view(3, 32, 224, 224)
        features_ho = torch.tensor(data['features'][:,-2:], dtype=torch.float32) # remove the gaze features
        label = torch.tensor(data['label'], dtype=torch.long)

        gaze = np.load(self.gaze_files[idx])
        gaze = torch.tensor(gaze, dtype=torch.float32)

        features = torch.cat((gaze,features_ho), dim=1)

        return {
            'images': images,
            'features': features,
            'label': label
        }
    

class PreprocessedImageGazeDataset(Dataset):
    """Load preprocessed image, gaze-hand-object features and label data."""
    def __init__(self, data_folder, gaze_folder, transform=None):
        self.data_folder = data_folder
        self.gaze_folder = gaze_folder
        self.transform = transform
        self.data_files = sorted(glob.glob(os.path.join(data_folder, '*.npz')))
        self.gaze_files = sorted(glob.glob(os.path.join(gaze_folder, '*.npy')))

        # Extract basenames without extensions
        self.data_basenames = [os.path.splitext(os.path.basename(f))[0] for f in self.data_files]
        self.gaze_basenames = [os.path.splitext(os.path.basename(f))[0] for f in self.gaze_files]

        if set(self.data_basenames) != set(self.gaze_basenames):
            raise ValueError("Mismatch between data files and gaze files. Ensure they have matching filenames.")

        # Create a mapping from basename to file path
        self.data_file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in self.data_files}
        self.gaze_file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in self.gaze_files}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_basename = self.data_basenames[idx]

        data_path = self.data_file_map[data_basename]
        gaze_path = self.gaze_file_map[data_basename]

        data = np.load(data_path)
        images = torch.tensor(data['images'], dtype=torch.float32)
        images = images / 255.0
        if self.transform:
            images = self.transform(images)
        images = images.view(3, 32, 224, 224)

        features_ho = torch.tensor(data['features'][:, -2:], dtype=torch.float32)  # Remove the gaze features
        label = torch.tensor(data['label'], dtype=torch.long)

        gaze = np.load(gaze_path)
        gaze = torch.tensor(gaze, dtype=torch.float32)

        features = torch.cat((gaze, features_ho), dim=1)

        return {
            'images': images,
            'features': features,
            'label': label
        }


class PreprocessedImageHODataset(Dataset):
    """Load preprocessed image, hand-object features and label data."""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.data_files = sorted(glob.glob(os.path.join(data_folder, '*.npz')))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        images = torch.tensor(data['images']).float()
        images = images / 255.0
        if self.transform:
            images = self.transform(images)
        # reshape the images to (3, 32, 224, 224)
        images = images.view(3, 32, 224, 224)
        # print(f"images shape is {images.shape}")
        features = torch.tensor(data['features'][:,-2:]) # Only use hand-object features
        # print(f"features shape is {features.shape}")
        label = torch.tensor(data['label'])

        # if self.transform:
        #     images = torch.stack([self.transform(image) for image in images])

        return {
            'images': images,
            'features': features,
            'label': label
        }


class PreprocessedOnlyGazeDataset(Dataset):
    """Load preprocessed image, gaze features and label data."""
    def __init__(self, data_folder, gaze_folder, transform=None):
        self.data_folder = data_folder
        self.gaze_folder = gaze_folder
        self.transform = transform
        self.data_files = sorted(glob.glob(os.path.join(data_folder, '*.npz')))
        self.gaze_files = sorted(glob.glob(os.path.join(gaze_folder, '*.npy')))

        # Extract basenames without extensions
        self.data_basenames = [os.path.splitext(os.path.basename(f))[0] for f in self.data_files]
        self.gaze_basenames = [os.path.splitext(os.path.basename(f))[0] for f in self.gaze_files]

        if set(self.data_basenames) != set(self.gaze_basenames):
            raise ValueError("Mismatch between data files and gaze files. Ensure they have matching filenames.")

        # Create a mapping from basename to file path
        self.data_file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in self.data_files}
        self.gaze_file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in self.gaze_files}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_basename = self.data_basenames[idx]

        data_path = self.data_file_map[data_basename]
        gaze_path = self.gaze_file_map[data_basename]

        data = np.load(data_path)
        images = torch.tensor(data['images'], dtype=torch.float32)
        images = images / 255.0
        if self.transform:
            images = self.transform(images)
        images = images.view(3, 32, 224, 224)

        label = torch.tensor(data['label'], dtype=torch.long)

        gaze = np.load(gaze_path)
        gaze = torch.tensor(gaze, dtype=torch.float32)
        features = gaze

        return {
            'images': images,
            'features': features,
            'label': label
        }
    

class PreprocessedOnlyImageDataset(Dataset):
    """Load the images and labels from the dataset"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.data_files = sorted(glob.glob(os.path.join(data_folder, '*.npz')))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        images = torch.tensor(data['images']).float()
        if self.transform:
            images = self.transform(images)
        # images = images.view(3, 32, 224, 224)

        label = torch.tensor(data['label'])

        return {
            'images':images, 
            'label':label
        }
    
