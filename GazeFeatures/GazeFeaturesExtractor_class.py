import sys
import numpy as np
import torch
import cv2
import os
import pandas as pd
import json
from GazeNet import GazeNet
import time


class GazeFeaturesExtractor:
    def __init__(self, frames_dir="/scratch/users/lu/msc2024_mengze/Frames",
                 gaze_data_dir="gaze_preprocessing/ExtractedGaze", output_dir="GazeFeatures", test_dir=""):
        self.frames_dir = frames_dir
        self.gaze_data_dir = gaze_data_dir
        self.output_dir = output_dir
        self.test_dir = test_dir  # only for testing
        self.width = 150
        self.height = 150
        self.gaze_feature_generator = GazeNet().cuda()

    @staticmethod
    def build_bounding_box(x, y, W, H):
        top_left_x = max(0, int(x - (W / 2)))
        top_left_y = max(0, int(y - (H / 2)))
        bottom_right_x = min(640, int(x + (W / 2)))
        bottom_right_y = min(480, int(y + (H / 2)))
        return top_left_x, top_left_y, bottom_right_x, bottom_right_y

    @staticmethod
    def transform_coordinate(x, y):
        x = x * 640
        y = y * 480
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
        cv2.imwrite(output_file_path, cropped_frame)

    def get_frame_list(self):
        file_list = []
        # os.path.join(self.frames_dir, self.test_dir) only for testing purpose
        for root, _, files in os.walk(os.path.join(self.frames_dir, self.test_dir)):
            for file in files:
                if file.endswith(".jpg"):
                    file_list.append(file)
        return file_list

    @staticmethod
    def parse_frame_name(frame_name):
        frame_name_parts, frame_number = frame_name.split("_")
        frame_number = int(frame_number.split(".")[0])
        frame_name_parts = frame_name_parts.split("-")
        video_name = '-'.join(frame_name_parts[:3])
        return video_name, frame_number

    def read_gaze_info(self, video_name, frame_number):
        """
        Reads the gaze information from a TXT file for a specific video and frame number.

        Args:
            video_name (str): The name of the video.
            frame_number (int): The frame number for which to retrieve the gaze information.

        Returns:
            tuple: A tuple containing the x-coordinate and y-coordinate of the gaze information.

        Raises:
            ValueError: If the file path is incorrect.

        """
        file_path = os.path.join(self.gaze_data_dir, f"{video_name}_processed.txt")
        if not os.path.exists(file_path):
            raise ValueError("Error: The file path is incorrect.")
        # Load the gaze information from the TXT file as a pandaframe
        gaze_info = pd.read_csv(file_path, sep="\t")
        inds = gaze_info.index[gaze_info['Frame'] == frame_number].tolist()

        if len(inds) == 0:
            # use the next frame until find the gaze information
            while len(inds) == 0:
                frame_number += 1
                inds = gaze_info.index[gaze_info['Frame'] == frame_number].tolist()
                # print("used next frame")

        # read the x_cor and y_cor from the gaze_info
        # if there are multiple frames with the same frame number
        x_cor, y_cor = gaze_info.loc[inds, ['xCoord', 'yCoord']].values[0]

        return x_cor, y_cor

    def save_gaze_feature(self, gaze_feature, image_name):
        # remove the .jpg from the image_name and add .json
        image_name = image_name.split(".")[0] + ".json"
        path_to_save = 'Extracted_HOFeatures/gaze/' + image_name
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

    def process_frames(self, save_cropped_frames=False):
        """
            Cropp the frames in 150x150 px and send in GazeNet() to get the gaze features(2048).

            Args:
                save_cropped_frames (bool, optional): Whether to save the cropped frames. Defaults to False.
            """
        countor = 0
        frames_list = self.get_frame_list()
        for frame in frames_list:
            video_name, frame_number = self.parse_frame_name(frame)
            frame_path = os.path.join(self.frames_dir, video_name, frame.split("_")[0], frame)
            x_cor, y_cor = self.read_gaze_info(video_name, frame_number)
            x_cor, y_cor = self.transform_coordinate(x_cor, y_cor)
            cropped_image = self.load_and_crop_frame(frame_path, x_cor, y_cor)

            # Save the cropped frame if needed
            if save_cropped_frames:
                self.save_cropped_frame(cropped_image, os.path.join(self.output_dir, "cropped_frames", frame))

            # change the type of cropped_image to tensor
            cropped_image = torch.from_numpy(cropped_image)
            # change the shape of cropped_image to (Bachsize, Channel, Width, Height)
            cropped_image = cropped_image.permute(2, 1, 0)
            cropped_image = cropped_image.unsqueeze(0)
            cropped_image = cropped_image.float()

            # Pad the cropped image to size 150x150
            cropped_image = torch.nn.functional.pad(cropped_image,
                                                    (0, 150 - cropped_image.shape[3], 0, 150 - cropped_image.shape[2]))

            # ship the cropped image to GPU
            cropped_image = cropped_image.cuda()

            gaze_feature_generator = GazeNet()
            # print(gaze_feature_generator)
            # ship the gaze_feature_generator to GPU
            gaze_feature_generator.cuda()
            gaze_feature = gaze_feature_generator(cropped_image)
            print(f"frame: {frame}")
            # Save the gaze feature
            self.save_gaze_feature(gaze_feature, frame)

            sys.stdout.write(f"\r====>Processed {countor} of {len(frames_list)} frames\n")
            sys.stdout.flush()

            countor += 1
            if countor == 32:
                break

    def concatenate_features(self):
        # get all gaze files in gaze, hand, object folder
        gaze_files = os.listdir("Extracted_HOFeatures/gaze")
        hand_files = os.listdir("Extracted_HOFeatures/hand")
        object_files = os.listdir("Extracted_HOFeatures/object")

        # create empty panda DataFrame with 2048 columns
        # columns = [i for i in range(2048)]
        # combined_features = pd.DataFrame(columns=columns)

        for file in gaze_files:
            combined_features = None
            feature_gaze = self.open_json("Extracted_HOFeatures/gaze/" + file)
            gaze_features_loaded = np.array(feature_gaze["gaze_features"])
            combined_features = gaze_features_loaded
            print(f"combined_features: {combined_features.shape}")

            if file in hand_files:
                feature_hand = self.open_json("Extracted_HOFeatures/hand/" + file)
                hand_features_loaded = np.array(feature_hand["hand_features"])
                combined_features = np.concatenate((combined_features, hand_features_loaded), axis=0)
                print(f"combined_features shape is {combined_features.shape}")

            if file in object_files:
                feature_object = self.open_json("Extracted_HOFeatures/object/" + file)
                object_features_loaded = np.array(feature_object["object_features"])
                combined_features = np.concatenate((combined_features, object_features_loaded), axis=0)
                print(f"combined_features shape is {combined_features.shape}")

            # save the combined_features as a pandaFrame file
            combined_features = pd.DataFrame(combined_features)
            print(combined_features.shape)
            print(combined_features.head())
            combined_features.to_csv("Extracted_HOFeatures/combined_features/" + file.split(".")[0] + ".csv",
                                     index=False)

            # # save the combined_features in txt file
            # with open("Extracted_HOFeatures/combined_features/" + file.split(".")[0] + ".txt", 'w') as f:
            #     for item in combined_features:
            #         f.write("%s\n" % item)
            # print(f"combined_features: {combined_features.shape}")
            # # print(f"combined_features: {combined_features}")
        pass


# Example usage
extractor = GazeFeaturesExtractor(test_dir="OP02-R07-Pizza/OP02-R07-Pizza-29330-30566-F000700-F000737")
# extractor.process_frames()

# print(os.getcwd())
extractor.concatenate_features()
