# -------------------------------------------------------------------
# Rewrite the dataset class to load the image, gaze and label data
# Three Classes: ImageGazeDataset, ImageHODataset, OnlyGazeDataset
# Update from myDataset3.py and myDataset4.py
# -------------------------------------------------------------------

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


class ImageGazeDataset(Dataset):
    """Load image, gaze-hand-object features and label data."""
    def __init__(self, image_folder, gaze_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.gaze_folder = gaze_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_pf = pd.read_csv(self.label_folder, sep=' ', header=None)
        self.clips = sorted(glob.glob(os.path.join(image_folder, '*')))
        # self.clips = [clip for clip in self.clips if len(glob.glob(os.path.join(os.path.dirname(clip), '*.jpg'))) == 32]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        image_paths = sorted(glob.glob(os.path.join(clip_path, '*.jpg')))
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        # base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        features = []

        # Load image
        # to cuda
        # image_s = time.time()
        images = [read_image(image_path) for image_path in image_paths]
        # to train.py
        # if self.transform:
        #     images = [self.transform(image) for image in images]
        # Shape should be (32, 3, H, W)
        images_tensor = torch.stack(images)
        # image_e = time.time()
        # print(f"load image time is {image_e - image_s}")

        # Load gaze data
        # feature_s = time.time()
        base_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]
        for base_name in base_names:
            csv_path = os.path.join(self.gaze_folder, base_name + '.csv')
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            # print(f"data is {data}")
            # print(f"data shape is {data.shape}")
            data = torch.tensor(data, dtype=torch.float32)
            features.append(data)
        # stack all gaze tensors for this clip 
        features = torch.stack(features)
        # feature_e = time.time()
        # print(f"load gaze time is {feature_e - feature_s}")

        # Load label data
        # label_s = time.time()
        label = self.label_pf[self.label_pf[0] == clip_name][1].values[0]
        # change the number from 1-106 to 0-105 to pass the index of scroes
        label -= 1
        label = torch.tensor(label, dtype=torch.long)
        # label_e = time.time()
        # print(f"load label time is {label_e - label_s}")
        # print(f"clip_name is {clip_name} and label is {label}")
        return {
            'images': images_tensor,
            'features': features,
            'label': label
        }


class ImageHODataset(Dataset):
    """Load image, hand-object features and label data."""
    def __init__(self, image_folder, gaze_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.gaze_folder = gaze_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_pf = pd.read_csv(self.label_folder, sep=' ', header=None)
        self.clips = sorted(glob.glob(os.path.join(image_folder, '*')))
        # self.clips = [clip for clip in self.clips if len(glob.glob(os.path.join(os.path.dirname(clip), '*.jpg'))) == 32]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        image_paths = sorted(glob.glob(os.path.join(clip_path, '*.jpg')))
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        
        # Load image
        # image_s = time.time()
        images = [read_image(image_path).to('cuda') for image_path in image_paths]
        # to train.py
        # if self.transform:
        #     images = [self.transform(image) for image in images]
        # Shape should be (32, 3, H, W)
        images_tensor = torch.stack(images)
        # image_e = time.time()
        # print(f"load image time is {image_e - image_s}")

        # Load gaze data
        features = []
        base_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]
        for base_name in base_names:
            csv_path = os.path.join(self.gaze_folder, base_name + '.csv')
            data = np.loadtxt(csv_path, delimiter=',', skiprows=2)
            # print(f"data shape is {data.shape}")
            data = torch.tensor(data, dtype=torch.float32, device='cuda')
            features.append(data)
        # stack all gaze tensors for this clip 
        features = torch.stack(features)

        # Load label data
        # label_s = time.time()
        label = self.label_pf[self.label_pf[0] == clip_name][1].values[0]
        # change the number from 1-106 to 0-105 to pass the index of scroes
        label -= 1
        label = torch.tensor(label, dtype=torch.long, device='cuda')
        # label_e = time.time()
        # print(f"load label time is {label_e - label_s}")
        # print(f"clip_name is {clip_name} and label is {label}")
        return {
            'images': images_tensor,
            'features': features,
            'label': label
        }


class OnlyGazeDataset(Dataset):
    """Load image, gaze features and label data."""
    def __init__(self, image_folder, gaze_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.gaze_folder = gaze_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_pf = pd.read_csv(self.label_folder, sep=' ', header=None)
        self.clips = sorted(glob.glob(os.path.join(image_folder, '*')))
        # self.clips = [clip for clip in self.clips if len(glob.glob(os.path.join(os.path.dirname(clip), '*.jpg'))) == 32]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        image_paths = sorted(glob.glob(os.path.join(clip_path, '*.jpg')))
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        
        # Load image
        # image_s = time.time()
        images = [read_image(image_path).to('cuda') for image_path in image_paths]
        # to train.py
        # if self.transform:
        #     images = [self.transform(image) for image in images]
        # Shape should be (32, 3, H, W)
        images_tensor = torch.stack(images)
        # image_e = time.time()
        # print(f"load image time is {image_e - image_s}")

        # Load gaze data
        features = []
        base_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]
        for base_name in base_names:
            csv_path = os.path.join(self.gaze_folder, base_name + '.csv')
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1, max_rows=1)
            print(f"data shape is {data.shape}")
            data = torch.tensor(data, dtype=torch.float32, device='cuda')
            features.append(data)
        # stack all gaze tensors for this clip 
        features = torch.stack(features)

        # Load label data
        # label_s = time.time()
        label = self.label_pf[self.label_pf[0] == clip_name][1].values[0]
        # change the number from 1-106 to 0-105 to pass the index of scroes
        label -= 1
        label = torch.tensor(label, dtype=torch.long, device='cuda')
        # label_e = time.time()
        # print(f"load label time is {label_e - label_s}")
        # print(f"clip_name is {clip_name} and label is {label}")
        return {
            'images': images_tensor,
            'features': features,
            'label': label
        }
    

class MulDataset(Dataset):
    """Use ThreadPoolExecutor to accelerate the data loading process."""
    def __init__(self, image_folder, gaze_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.gaze_folder = gaze_folder
        self.transform = transform
        self.label_pf = pd.read_csv(label_folder, sep=' ', header=None)
        self.clips = sorted(glob.glob(os.path.join(image_folder, '*')))
        self.label_map = {row[0]: row[1] - 1 for _, row in self.label_pf.iterrows()}  # Preload labels and adjust index

    def __len__(self):
        return len(self.clips)

    def load_images(self, image_paths):
        return [read_image(image_path) for image_path in image_paths]

    def load_gaze_data(self, csv_path):
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        return torch.tensor(data, dtype=torch.float32)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        image_paths = sorted(glob.glob(os.path.join(clip_path, '*.jpg')))
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]

        with ThreadPoolExecutor() as executor:
            images_future = executor.submit(self.load_images, image_paths)
            base_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]
            gaze_futures = [executor.submit(self.load_gaze_data, os.path.join(self.gaze_folder, base_name + '.csv')) for base_name in base_names]

            images = images_future.result()
            features = [gaze_future.result() for gaze_future in gaze_futures]

        images_tensor = torch.stack(images)
        features = torch.stack(features)
        label = torch.tensor(self.label_map[clip_name], dtype=torch.long)

        return {
            'images': images_tensor,
            'features': features,
            'label': label
        }
    

# # 假定文件夹路径
# image_folder = '/scratch/users/lu/msc2024_mengze/Frames3/test_split1'
# gaze_folder = '/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures/test_split1/combined_features_new'
# label_folder = '/scratch/users/lu/msc2024_mengze/dataset/test_split12.txt'

# # 图像预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # 创建数据集实例
# dataset = ImageGazeDataset(image_folder, gaze_folder, label_folder, transform=None)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for data in dataloader:
#     images, gazes, labels = data['image'], data['gaze'], data['label']
#     # print(f"images shape is {images.shape}")
#     # print(f"gaze shape is {gazes.shape}")
#     # print(f"label shape is {labels.shape}")
# # 保存数据集
# # dataset.save_dataset()