# Data preprocessing script for the Transformer model
# This script reads the image frames, gaze-hand-object features and labels,
# Then saves them into a single .npz file for each video clip

import os
import glob
import numpy as np
import pandas as pd
from torchvision.io import read_image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def preprocess_data(image_folder, gaze_folder, label_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    label_df = pd.read_csv(label_file, sep=' ', header=None)
    label_map = {row[0]: row[1] - 1 for _, row in label_df.iterrows()}  # Preload labels and adjust index

    clips = sorted(glob.glob(os.path.join(image_folder, '*')))

    def process_clip(clip_path):
        image_paths = sorted(glob.glob(os.path.join(clip_path, '*.jpg')))
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        base_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]

        # Load images
        images = [read_image(image_path).numpy() for image_path in image_paths]

        # Load features
        features = []
        for base_name in base_names:
            # csv_path = os.path.join(gaze_folder, base_name + '.csv')
            npy_path = os.path.join(gaze_folder, base_name + '.npy')
            # data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            data = np.load(npy_path)
            features.append(data)
        features = np.stack(features)

        # Get label
        label = int(label_map[clip_name])

        # Save to a single file
        np.savez(os.path.join(output_folder, clip_name + '.npz'), images=images, features=features, label=label)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_clip, clip): clip for clip in clips}
        for future in tqdm(as_completed(futures), total=len(clips)):
            future.result()  # Ensure any exceptions are raised

# Usage
os.chdir("/scratch/users/lu/msc2024_mengze/")
image_folder = 'Frames_224/train_split1'
feature_folder = 'Extracted_Features_224/train_split1/gaze'
label_file = 'dataset/train_split12.txt'
output_folder = 'Extracted_Features_224/train_split1/only_gaze_features'
preprocess_data(image_folder, feature_folder, label_file, output_folder)
