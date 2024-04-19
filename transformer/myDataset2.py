import os
import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch
import pandas as pd

class ImageGazeDataset(Dataset):
    def __init__(self, image_folder, gaze_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.gaze_folder = gaze_folder
        self.label_folder = label_folder
        self.transform = transform
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
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        if self.transform:
            images = [self.transform(image) for image in images]
        # Shape should be (32, 3, H, W)
        images_tensor = torch.stack(images)  

        # Load gaze data
        base_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]
        for base_name in base_names:
            csv_path = os.path.join(self.gaze_folder, base_name + '.csv')
            df = pd.read_csv(csv_path)
            if df.shape[0] > 5:
                df = df.head(5)
            features.append(torch.tensor(df.values, dtype=torch.float32))
        # stack all gaze tensors for this clip 
        features = torch.stack(features)

        # Load label data
        label_pf = pd.read_csv(self.label_folder, sep=' ', header=None)
        label = label_pf[label_pf[0] == clip_name][1].values[0]
        label = torch.tensor(label, dtype=torch.long)

        return {
            'images': images_tensor,
            'features': features,
            'label': label
        }
# Assuming transforms and paths initialization are correct as per the user's environment§


# # 假定文件夹路径
# image_folder = '/scratch/users/lu/msc2024_mengze/Frames3/test_use'
# gaze_folder = '/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures/test_split1/combined_features'
# label_folder = '/scratch/users/lu/msc2024_mengze/dataset/test_split12.txt'

# # 图像预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # 创建数据集实例
# dataset = ImageGazeDataset(image_folder, gaze_folder, label_folder, transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for data in dataloader:
#     images, gazes, labels = data['image'], data['gaze'], data['label']
#     # print(f"images shape is {images.shape}")
#     # print(f"gaze shape is {gazes.shape}")
#     # print(f"label shape is {labels.shape}")
# # 保存数据集
# # dataset.save_dataset()