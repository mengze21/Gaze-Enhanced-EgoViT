# ----------------------------------------------------------------
# update from GazeFeatures/GazeFeaturesExtractor_setseed.py
# The weights of GazeNet() is fixed, and the seed is set to 42
# Lack of the gaze data is filled with the center of the frame
# The new version 05/12
# ----------------------------------------------------------------

import sys
import numpy as np
import torch
import cv2
import os
import pandas as pd
import json
from GazeNet import GazeNet_const_weight
import time
import glob


class GazeFeaturesExtractor:
    def __init__(self, frames_dir="/scratch/users/lu/msc2024_mengze/Frames3/testing",
                 gaze_data_dir="/scratch/users/lu/msc2024_mengze/gaze_preprocessing/ExtractedGaze",
                 output_dir="/scratch/users/lu/msc2024_mengze/GazeFeatures",
                 gazepath_tosave = "Extracted_HOFeatures/testing/gaze",
                 bbox_hw=35, test_dir=""):
        self.frames_dir = frames_dir
        self.gaze_data_dir = gaze_data_dir
        self.output_dir = output_dir
        self.test_dir = test_dir  # only for testing
        self.width = bbox_hw
        self.height = bbox_hw
        self.gazepath_tosave = gazepath_tosave
        self.gaze_feature_generator = GazeNet_const_weight()

    @staticmethod
    def build_bounding_box(x, y, W, H):
        top_left_x = max(0, int(x - (W / 2)))
        top_left_y = max(0, int(y - (H / 2)))
        bottom_right_x = min(640, int(x + (W / 2)))
        bottom_right_y = min(480, int(y + (H / 2)))
        return top_left_x, top_left_y, bottom_right_x, bottom_right_y

    @staticmethod
    def transform_coordinate(x, y):
        x = x * 160
        y = y * 120
        return int(x), int(y)

    def load_and_crop_frame(self, file_path, center_x, center_y):
        frame = cv2.imread(file_path)
        # check if the frame is None
        if frame is None:
            raise ValueError(f"Error: The frame is None. The file path is {file_path}")

        x1, y1, x2, y2 = self.build_bounding_box(center_x, center_y, self.width, self.height)
        cropped_frame = frame[y1:y2, x1:x2]
        return cropped_frame

    @staticmethod
    def save_cropped_frame(cropped_frame, output_file_path):
        if not os.path.exists(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))
        # print(f"output_file_path: {output_file_path}")
        # print(f"output_file_path: {output_file_path}")
        cv2.imwrite(output_file_path, cropped_frame)

    def get_frame_list(self):
        file_list = []
        # print(f"---self.frames_dir: {self.frames_dir}")
        # print(f"---self.test_dir: {self.test_dir}")
        # os.path.join(self.frames_dir, self.test_dir) only for testing purpose
        for root, _, files in os.walk(os.path.join(self.frames_dir, self.test_dir)):
            for file in files:
                if file.endswith(".jpg"):
                    file_list.append(file)
        return file_list
    
    def get_video_list(self):
        video_names = glob.glob(os.path.join(self.frames_dir, '*'))

        return video_names
    
    @staticmethod
    def parse_frame_name(frame_name):
        # frame_name_parts, frame_number = frame_name.split("_")
        parts = frame_name.split("_")
        frame_name_parts = parts[0]
        frame_number = parts[1]
        frame_name_parts = frame_name_parts.split("-")
        video_name = '-'.join(frame_name_parts[:3])
        frame_number = frame_number.split("_")[0]
        frame_number = int(frame_number.split(".")[0])
        return video_name, frame_number

    def read_gaze_info(self, video_name, frame_number):
        """
        Reads the gaze information from a TXT file for a specific video and frame number.
        Lack of the gaze data is filled with the center of the frame.

        Args:
            video_name (str): The name of the video.
            frame_number (int): The frame number for which to retrieve the gaze information.

        Returns:
            tuple: A tuple containing the x-coordinate and y-coordinate of the gaze information.

        Raises:
            ValueError: If the file path is incorrect.

        """
        center_gaze = False
        # print(f"---video_name: {video_name}, frame_number: {frame_number}")
        # print(f"---self.gaze_data_dir: {self.gaze_data_dir}")
        # print(f"current dir: {os.getcwd()}")
        file_path = os.path.join(self.gaze_data_dir, f"{video_name}_processed.txt")
        # print(f"---file_path: {file_path}")
        if not os.path.exists(file_path):
            raise ValueError("Error: The file path is incorrect.")
        # Load the gaze information from the TXT file as a pandaframe
        gaze_info = pd.read_csv(file_path, sep="\t")
        
        inds = gaze_info.index[gaze_info['Frame'] == frame_number].tolist()
        # print(f"inds: {inds}")
        if len(inds) == 0:
            counter = 0
            # use the next frame until find the gaze information
            while len(inds) == 0 and center_gaze is False:
                frame_number += 1
                inds = gaze_info.index[gaze_info['Frame'] == frame_number].tolist()
                counter += 1
                if counter == 20:
                    center_gaze = True
                # print(f"use the next frame: {frame_number}")

        if center_gaze:
            print("No gaze data found, use center point as gaze data")
            # center of the frame (160, 120) is (80, 60)
            # Normalize the center coordinates
            x_cor = 0.5
            y_cor = 0.5
        else:
            # read the x_cor and y_cor from the gaze_info
            # if there are multiple frames with the same frame number
            # print(f"inds: {inds}")
            x_cor, y_cor = gaze_info.loc[inds, ['xCoord', 'yCoord']].values[0]

        return x_cor, y_cor

    def save_gaze_feature(self, gaze_feature, image_name):
        # remove the .jpg from the image_name and add .json
        image_name = image_name.split(".")[0] + ".json"
        # print(f"image_name: {image_name}")
        # dir_path = os.path.join("Extracted_HOFeatures/test_split1", "gaze_new")
        # path_to_save = '/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures_test/gaze/' + image_name
        path_to_save = os.path.join(self.gazepath_tosave, image_name)
        if not os.path.exists(self.gazepath_tosave):
            os.makedirs(self.gazepath_tosave)

        combined_data = {
            "class": "gaze",
            "image_name": image_name,
            "gaze_features": gaze_feature.tolist()
        }
        with open(path_to_save, 'w') as f:
            json.dump(combined_data, f, indent=4)

    def open_json(self, file_path):
        with open(file_path, 'r') as f:
            data_loaded = json.load(f)

        return data_loaded

    def padding_feature(self, feature):
        """
        Pad the feature to [3, 2048] using zero padding.
        Args:
            feature (numpy.ndarray): The feature to be padded.
        Returns:
            numpy.ndarray: The padded feature array with shape [3, 2048].
        """
        # Calculate the number of rows to pad
        padding_rows = 3 - feature.shape[0]
        # Create a tensor of zeros with the required padding rows
        padd_data = np.zeros((padding_rows, 2048))
        # Concatenate the original feature tensor with the padding tensor
        feature = np.concatenate((feature, padd_data), axis=0)
        # for i in range(3 - len(feature)):
        #     # random data between 0 and 1
        #     padd_data = torch.zeros(1, 2048)
        #     feature = torch.cat((feature, padd_data), dim=0)

        return feature

    def set_seed(self, seed):
        """设置固定的随机种子以确保可重复性。"""
        torch.manual_seed(seed)  # 为CPU设置种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # 为所有GPU设置种子
            # torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，为所有GPU设置种子
        torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的，如果不设置的话，可能会因为选择的算法不同而导致每次网络前馈时结果略有差异
        torch.backends.cudnn.benchmark = False  # 当网络结构不变时，关闭优化，提高可复现性

    def process_frames(self, save_cropped_frames=False):
        """
            Cropp the frames in 150x150 px and send in GazeNet() to get the gaze features(2048).

            Args:
                save_cropped_frames (bool, optional): Whether to save the cropped frames. Defaults to False.
            """
        countor = 0
        frames_list = self.get_frame_list()
        print(f"frmames_list length: {len(frames_list)}")
        for frame in frames_list:
            video_name, frame_number = self.parse_frame_name(frame)
            frame_path = os.path.join(self.frames_dir, frame.split("_")[0], frame)
            # print(f"frame_path: -------{frame_path}")
            x_cor, y_cor = self.read_gaze_info(video_name, frame_number)
            x_cor, y_cor = self.transform_coordinate(x_cor, y_cor)

            # print(f"x_cor: {x_cor}, y_cor: {y_cor}")
            cropped_image = self.load_and_crop_frame(frame_path, x_cor, y_cor)

            # Save the cropped frame if needed
            if save_cropped_frames:
                self.save_cropped_frame(cropped_image, os.path.join(self.output_dir, "cropped_frames2", frame))

            # change the type of cropped_image to tensor
            cropped_image = torch.from_numpy(cropped_image)
            # change the shape of cropped_image to (Bachsize, Channel, Width, Height)
            cropped_image = cropped_image.permute(2, 1, 0).contiguous()
            cropped_image = cropped_image.unsqueeze(0)
            cropped_image = cropped_image.float()

            # Pad the cropped image to size bbox_hw x bbox_hw
            cropped_image = torch.nn.functional.pad(cropped_image,
                                                    (0, self.width - cropped_image.shape[3], 0, self.height - cropped_image.shape[2]))
            # print(f"cropped_image shape: {cropped_image.shape}")
            # ship the cropped image to GPU
            cropped_image = cropped_image.cuda()

            self.set_seed(42)
            with torch.no_grad():
                gaze_feature_generator = GazeNet_const_weight()
                # print(gaze_feature_generator)
                # ship the gaze_feature_generator to GPU
                gaze_feature_generator.cuda()
                gaze_feature = gaze_feature_generator(cropped_image)
            # print(f"frame: {frame}")
            # Save the gaze feature
            self.save_gaze_feature(gaze_feature, frame)

            sys.stdout.write(f"\r====>Processed {countor} of {len(frames_list)} frames\n")
            sys.stdout.flush()

            countor += 1
            # if countor == 32:
            #     break

    def concatenate_features(self):
        # gaze_folder = os.path.join(folder, "gaze")
        # hand_folder = os.path.join(folder, "hand")
        # object_folder = os.path.join(folder, "object")
        # get all gaze files in gaze, hand, object folder
        feature_dir = self.gazepath_tosave.replace('/gaze', '')
        gaze_path = os.path.join(feature_dir, "gaze")
        hand_path = os.path.join(feature_dir, "hand")
        object_path = os.path.join(feature_dir, "object")
        combined_features_path = os.path.join(feature_dir, "combined_features")

        gaze_files = os.listdir(gaze_path)
        hand_files = os.listdir(hand_path)
        object_files = os.listdir(object_path)
        # gaze_files = os.listdir(gaze_folder)
        # hand_files = os.listdir(hand_folder)
        # object_files = os.listdir(object_folder)

        # create empty panda DataFrame with 2048 columns
        # columns = [i for i in range(2048)]
        # combined_features = pd.DataFrame(columns=columns)
        print("====>Start to concatenate features")

        for file in gaze_files:
            combined_features = None
            feature_gaze = self.open_json(gaze_path + '/' + file)
            gaze_features_loaded = np.array(feature_gaze["gaze_features"])
            combined_features = gaze_features_loaded
            # print(f"combined_features: {combined_features.shape}")

            if file in hand_files:
                feature_hand = self.open_json(hand_path + '/' + file)
                hand_features_loaded = np.array(feature_hand["hand_features"])
                # print(f"hand_features_loaded shape is {hand_features_loaded.shape}")

                # apply average pooling to hand_features_loaded to make it (1, 2048)
                hand_features_avg = np.mean(hand_features_loaded, axis=0, keepdims=True)
                # print(f"hand_features_avg shape is {hand_features_avg.shape}")
                combined_features = np.concatenate((combined_features, hand_features_avg), axis=0)
                # print(f"combined_features shape is {combined_features.shape}")

            if file in object_files:
                feature_object = self.open_json(object_path + '/' + file)
                object_features_loaded = np.array(feature_object["object_features"])
                # print(f"object_features_loaded shape is {object_features_loaded.shape}")

                # apply average pooling to object_features_loaded to make it (1, 2048)
                object_features_avg = np.mean(object_features_loaded, axis=0, keepdims=True)
                # print(f"object_features_avg shape is {object_features_avg.shape}")
                combined_features = np.concatenate((combined_features, object_features_avg), axis=0)
                # print(f"combined_features shape is {combined_features.shape}")

            # pad the combined_features to (3, 2048)
            # print(f"len(combined_features): {len(combined_features)}")
            if len(combined_features) < 3:
                combined_features = self.padding_feature(combined_features)

            # combined_features has the shape of (3, 2048), row 1 is gaze, row 2 is hand, row 3 is object
            # save the combined_features as a pandaFrame file
            combined_features = pd.DataFrame(combined_features, index=None)
            # change to float32
            combined_features = combined_features.astype(np.float32)



            # save the combined_features in csv file
            combined_features.to_csv(combined_features_path + '/' + file.split(".")[0] + ".csv",
                                     index=False)

            # # save the combined_features in txt file
            # with open("Extracted_HOFeatures/combined_features/" + file.split(".")[0] + ".txt", 'w') as f:
            #     for item in combined_features:
            #         f.write("%s\n" % item)
            # print(f"combined_features: {combined_features.shape}")
            # # print(f"combined_features: {combined_features}")
        print("====>Concatenated all features")


# Example usage
# change the working dir to msc2024_mengze
os.chdir("/scratch/users/lu/msc2024_mengze")
# extractor = GazeFeaturesExtractor(frames_dir="/scratch/users/lu/msc2024_mengze/Frames3/train_split1",test_dir="OP03-R01-PastaSalad-155200-157750-F003718-F003792")

# Extract gaze features
# start_time = time.time()
# extractor = GazeFeaturesExtractor(frames_dir="Frames3/train_split1", gazepath_tosave="Extracted_HOFeatures/train_split1/gaze")
# # videos = extractor.get_video_list()
# # print(f"videos: {videos[0]}")
# extractor.process_frames(save_cropped_frames=False)
# end_time = time.time()
# print(f"---Time to extract gaze features: {(end_time - start_time) / 60:.2f} minutes---")

# Concatenate features
start_time2 = time.time()
extractor = GazeFeaturesExtractor(frames_dir="Frames3/train_split1", gazepath_tosave="Extracted_HOFeatures/train_split1/gaze")
extractor.concatenate_features()
end_time2 = time.time()
print(f"---Time to concatenate features: {(end_time2 - start_time2) / 60:.2f} minutes---")