import os
import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch
import pandas as pd

class ImageGazeDataset(Dataset):
    def __init__(self, image_folder, gaze_folder, label_folder, transform=None):
        self.clips = {}
        self.transform = transform
        
        clip_paths = glob.glob(os.path.join(image_folder, '*'))
        
        for clip_path in clip_paths:
            clip_name = os.path.basename(clip_path)
            image_paths = sorted(glob.glob(os.path.join(clip_path, '*.jpg')))
            if len(image_paths) != 32:
                # Skip any clip that does not have exactly 32 frames
                continue  

            gaze_data = []
            labels = []
            for path in image_paths:
                base_name = os.path.splitext(os.path.basename(path))[0]

                # Load gaze data
                csv_path = os.path.join(gaze_folder, base_name + '.csv')
                df = pd.read_csv(csv_path)
                # if df has more than 5 rows, not include head keep first 5 rows
                if df.shape[0] > 5:
                    df = df.head(5)
                gaze_data.append(torch.tensor(df.values, dtype=torch.float32))
                
                # Load label data
                label_pf = pd.read_csv(label_folder, sep=' ', header=None)
                label = label_pf[label_pf[0] == base_name.split('_')[0]][1].values[0]
                # labels.append(torch.tensor(label, dtype=torch.int32))
            
            # Stack all gaze and label tensors for this clip
            self.clips[clip_name] = {
                'images': image_paths,
                'gaze': torch.stack(gaze_data),
                # 'label': torch.tensor(labels).unsqueeze(-1)  # Shape (32, 1)
                # 'label': torch.stack(labels) # Shape (32, 1)
                'label': torch.tensor(label, dtype=torch.long)
            }
            # print(f"images shape is {len(self.clips[clip_name]['images'])}")
            # print(f"gaze shape is {self.clips[clip_name]['gaze'].shape}")
            # print(f"label shape is {self.clips[clip_name]['label'].shape}")
            # print(f"label value is {self.clips[clip_name]['label']}")
            # print(f"label is {self.clips[clip_name]['label']}")

    def __len__(self):
        return len(self.clips)

    def save_dataset(self, save_path='/scratch/users/lu/msc2024_mengze/dataset/dataset_test_split1.pt'):
        """
        Saves the entire dataset as a single .pt file.

        Args:
            save_path (str): The path to save the dataset file.
        """
        all_samples = []
        for idx in range(len(self)):
            sample = self.__getitem__(idx)  # Call __getitem__ to get a sample
            all_samples.append(sample)

        torch.save(all_samples, save_path)
        print(f"Dataset saved to: {save_path}")

    def __getitem__(self, idx):
        clip_name = list(self.clips.keys())[idx]
        clip_data = self.clips[clip_name]

        images = [Image.open(path).convert('RGB') for path in clip_data['images']]
        if self.transform:
            images = [self.transform(image) for image in images]
        
        images_tensor = torch.stack(images)  # Shape should be (32, 3, H, W), H and W depend on the transform
        
        sample = {
            'images': images_tensor,
            'gaze': clip_data['gaze'],
            'label': clip_data['label']
        }
        return sample

# Assuming transforms and paths initialization are correct as per the user's environment


# # 假定文件夹路径
# image_folder = '/scratch/users/lu/msc2024_mengze/Frames3/test_split1'
# gaze_folder = '/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures/test_split1/combined_features'
# label_folder = '/scratch/users/lu/msc2024_mengze/dataset/test_split12.txt'

# # 图像预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # 创建数据集实例
# dataset = ImageGazeDataset(image_folder, gaze_folder, label_folder, transform=transform)

# 保存数据集
# dataset.save_dataset()