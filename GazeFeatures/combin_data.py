# --------------------------------------------------------------------------
# conbin frames, gaze, hand, object features and label in a single .npz file
# --------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
from torchvision.io import read_image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Set the current working directory
os.chdir('/scratch/users/lu/msc2024_mengze')

def combine_data(image_folder, gaze_folder, hand_folder, object_folder, label_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    label_df = pd.read_csv(label_file, sep=' ', header=None)
    label_map = {row[0]: row[1] - 1 for _, row in label_df.iterrows()}  # Preload labels and adjust index

    clips = sorted(glob.glob(os.path.join(image_folder, '*')))

    gaze_files = set(os.listdir(gaze_folder))
    hand_files = set(os.listdir(hand_folder))
    object_files = set(os.listdir(object_folder))

    def process_clip(clip_path):
        image_paths = sorted(glob.glob(os.path.join(clip_path, '*.jpg')))
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        base_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]

        # gaze_files = sorted(glob.glob(os.path.join(gaze_folder, '*.npy')))
        # hand_files = sorted(glob.glob(os.path.join(hand_folder, '*.npy')))
        # object_files = sorted(glob.glob(os.path.join(object_folder, '*.npy')))
        
        # Load images
        images = [read_image(image_path).numpy() for image_path in image_paths]

        # Load features
        gaze_features = []
        hand_features = []
        object_features = []

        for base_name in base_names:
            # load gaze features
            if base_name + '.npy' in gaze_files:
                gaze_fpth = os.path.join(gaze_folder, base_name + '.npy')
                data_g = np.load(gaze_fpth)
            else:
                data_g = np.zeros((1, 2048))  # zero data
            gaze_features.append(data_g)

            # load hand features
            if base_name + '.npy' in hand_files:
                hand_fpth = os.path.join(hand_folder, base_name + '.npy')
                data_h = np.load(hand_fpth)
                data_h = np.mean(data_h, axis=0, keepdims=True)
            else:
                data_h = np.zeros((1, 2048))
            hand_features.append(data_h)

            # load object features
            if base_name + '.npy' in object_files:
                object_fpth = os.path.join(object_folder, base_name + '.npy')
                data_o = np.load(object_fpth)
                data_o = np.mean(data_o, axis=0, keepdims=True)
            else:
                data_o = np.zeros((1, 2048))
            object_features.append(data_o)
        
        gaze_features = np.vstack(gaze_features)
        hand_features = np.vstack(hand_features)
        object_features = np.vstack(object_features)

        # Combine features for each frame to have shape [32, 3, 2048]
        features = np.stack((gaze_features, hand_features, object_features), axis=1)

        # Get label
        label = int(label_map[clip_name])

        # Save to a single file
        np.savez(os.path.join(output_folder, clip_name + '.npz'), images=images, features=features, label=label)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_clip, clip): clip for clip in clips}
        for future in tqdm(as_completed(futures), total=len(clips)):
            future.result()  # Ensure any exceptions are raised

def check_saved_data(clip):
    data = np.load(clip)
    print(f"clip: {os.path.basename(clip)}")
    print(data['images'].shape, data['features'].shape, data['label'])
    print(f"\nfeatures: {data['features'][0,:,:]}\n")

if __name__ == '__main__':
    image_folder = 'Frames_224/test_split1'
    gaze_folder = 'Extracted_Features_224/test_split1/gaze'
    hand_folder = 'Extracted_Features_224/test_split1/hand'
    object_folder = 'Extracted_Features_224/test_split1/object'
    label_file = 'dataset/test_split12.txt'
    output_folder = 'Extracted_Features_224/test_split1/all'

    combine_data(image_folder, gaze_folder, hand_folder, object_folder, label_file, output_folder)
    # check_saved_data(clip='Extracted_Features_224/train_split1/t/OP01-R01-PastaSalad-1115060-1121560-F026746-F026933.npz')