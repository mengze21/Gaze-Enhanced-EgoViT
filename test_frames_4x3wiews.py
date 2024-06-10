import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import os

# Custom resize to ensure only the shorter side is resized to 224
class ResizeShorterSide:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if h < w:
            new_h = self.size
            new_w = h
        else:
            new_w = self.size
            new_h = w
        return img.resize((new_w, new_h), Image.BILINEAR)

# Define transformations
class MultiViewTransform:
    def __init__(self):
        self.spatial_transform = ResizeShorterSide(224)

    def __call__(self, frames):
        clips = self.uniform_sample(frames)
        views = []
        for clip in clips:
            scaled_clip = [self.spatial_transform(F.to_pil_image(frame)) for frame in clip]
            scaled_clip = torch.stack([transforms.ToTensor()(img) for img in scaled_clip])
            crops = self.three_crops(scaled_clip)
            views.extend(crops)
        return torch.stack(views)

    def uniform_sample(self, frames):
        num_frames = len(frames)
        clip_len = num_frames // 4
        clips = [frames[i * clip_len: (i + 1) * clip_len] for i in range(4)]
        return clips

    def three_crops(self, clip):
        _, t, h, w = clip.shape
        center_l = (w - 224) // 2
        center_r = (w + 224) // 2
        crops = [
            clip[:, :, :, :224],  # Left crop
            clip[:, :, :, -224:],  # Right crop
            clip[:, :, :, center_l:center_r]  # Center crop
        ]
        return crops

# Custom dataset class
class VideoDataset(Dataset):
    def __init__(self, video_paths, transform=None):
        self.video_paths = video_paths
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self.load_frames(video_path)

        if self.transform:
            views = self.transform(frames)
        else:
            views = torch.stack([transforms.ToTensor()(frame) for frame in frames])

        video_name = os.path.basename(video_path).split('.')[0]
        return views.numpy(), video_name  # Return numpy array and video name

    def load_frames(self, video_path, num_frames=32):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break

        cap.release()

        # Padding the last frame if the total number of frames is less than num_frames
        if len(frames) < num_frames:
            last_frame = frames[-1]
            while len(frames) < num_frames:
                frames.append(last_frame)

        return frames

# Function to save views to npy file
def save_views_to_npy(video_paths, transform, save_dir):
    dataset = VideoDataset(video_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for views, video_name in dataloader:
        views = views.squeeze(0)  # Remove batch dimension
        save_path = os.path.join(save_dir, f'{video_name[0]}_views.npy')
        np.save(save_path, views)
        print(f"Saved {save_path}")

# Example usage
video_folder = '/scratch/users/lu/msc2024_mengze/gaze_dataset/cropped_clips'
video_paths = []
# get all video paths
for root, dirs, files in os.walk(video_folder):
    for file in files:
        if file.endswith('.mp4'):
            video_paths.append(os.path.join(root, file))

transform = MultiViewTransform()
save_dir = '/scratch/users/lu/msc2024_mengze/frames_4x3views'

save_views_to_npy(video_paths, transform, save_dir)