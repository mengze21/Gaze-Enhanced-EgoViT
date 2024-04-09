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
    def __init__(self, frames_dir="/scratch/users/lu/msc2024_mengze/Frames3/test_split1",
                 gaze_data_dir="/scratch/users/lu/msc2024_mengze/gaze_preprocessing/ExtractedGaze",
                 output_dir="/scratch/users/lu/msc2024_mengze/GazeFeatures",
                 gazepath_tosave = "/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures_test/gaze",
                 bbox_hw=35, test_dir=""):
        self.frames_dir = frames_dir
        self.gaze_data_dir = gaze_data_dir
        self.output_dir = output_dir
        self.test_dir = test_dir  # only for testing
        self.width = bbox_hw
        self.height = bbox_hw
        self.gazepath_tosave = gazepath_tosave
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

        Args:
            video_name (str): The name of the video.
            frame_number (int): The frame number for which to retrieve the gaze information.

        Returns:
            tuple: A tuple containing the x-coordinate and y-coordinate of the gaze information.

        Raises:
            ValueError: If the file path is incorrect.

        """
        random_gaze = False
        print(f"---video_name: {video_name}, frame_number: {frame_number}")
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
            while len(inds) == 0 and random_gaze is False:
                frame_number += 1
                inds = gaze_info.index[gaze_info['Frame'] == frame_number].tolist()
                counter += 1
                if counter == 20:
                    random_gaze = True
                print(f"use the next frame: {frame_number}")

        if random_gaze:
            print("No gaze information found, use random gaze information")
            # generate random gaze information value is between 0 and 1
            x_cor = np.random.rand()
            y_cor = np.random.rand()
        else:
            # read the x_cor and y_cor from the gaze_info
            # if there are multiple frames with the same frame number
            print(f"inds: {inds}")
            x_cor, y_cor = gaze_info.loc[inds, ['xCoord', 'yCoord']].values[0]

        return x_cor, y_cor

    def save_gaze_feature(self, gaze_feature, image_name):
        # remove the .jpg from the image_name and add .json
        image_name = image_name.split(".")[0] + ".json"
        # print(f"image_name: {image_name}")
        dir_path = os.path.join("Extracted_HOFeatures_train", "gaze")
        # path_to_save = '/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures_test/gaze/' + image_name
        path_to_save = os.path.join(dir_path, image_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        combined_data = {
            "class": "gaze",
            "image_name": image_name,
            "gaze_features": gaze_feature.tolist()
        }
        with open(path_to_save, 'w') as f:
            json.dump(combined_data, f, indent=4)

    def open_json(self, file_path):
        print(os.getcwd())
        with open(file_path, 'r') as f:
            data_loaded = json.load(f)

        return data_loaded

    def padding_feature(self, feature):
        """
        Pad the feature to the max_length
        Args:
            feature (list): The feature to be padded.
        Returns:
            list: The padded feature.
        """
        for i in range(5 - len(feature)):
            # random data between 0 and 1
            random_data = np.random.randn(1, 2048)
            # set the negative value to 0
            random_data[random_data < 0] = 0
            feature = np.concatenate((feature, random_data), axis=0)

        return feature

    def process_frames(self, save_cropped_frames=True):
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
            cropped_image = cropped_image.permute(2, 1, 0)
            cropped_image = cropped_image.unsqueeze(0)
            cropped_image = cropped_image.float()

            # Pad the cropped image to size bbox_hw x bbox_hw
            cropped_image = torch.nn.functional.pad(cropped_image,
                                                    (0, self.width - cropped_image.shape[3], 0, self.height - cropped_image.shape[2]))
            # print(f"cropped_image shape: {cropped_image.shape}")
            # ship the cropped image to GPU
            cropped_image = cropped_image.cuda()

            gaze_feature_generator = GazeNet()
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
        # get all gaze files in gaze, hand, object folder
        gaze_files = os.listdir("Extracted_HOFeatures_test/gaze")
        hand_files = os.listdir("Extracted_HOFeatures_test/hand")
        object_files = os.listdir("Extracted_HOFeatures_test/object")

        # create empty panda DataFrame with 2048 columns
        # columns = [i for i in range(2048)]
        # combined_features = pd.DataFrame(columns=columns)

        for file in gaze_files:
            combined_features = None
            feature_gaze = self.open_json("Extracted_HOFeatures_test/gaze/" + file)
            gaze_features_loaded = np.array(feature_gaze["gaze_features"])
            combined_features = gaze_features_loaded
            # print(f"combined_features: {combined_features.shape}")

            if file in hand_files:
                feature_hand = self.open_json("Extracted_HOFeatures_test/hand/" + file)
                hand_features_loaded = np.array(feature_hand["hand_features"])
                # if len(hand_features_loaded[0].shape) < 2:
                #     hand_features_loaded = self.padding_feature(hand_features_loaded)
                print(f"hand_features_loaded shape is {hand_features_loaded.shape}")
                combined_features = np.concatenate((combined_features, hand_features_loaded), axis=0)
                print(f"combined_features shape is {combined_features.shape}")

            if file in object_files:
                feature_object = self.open_json("Extracted_HOFeatures_test/object/" + file)
                object_features_loaded = np.array(feature_object["object_features"])
                # if len(object_features_loaded[0].shape) < 2:
                #     object_features_loaded = self.padding_feature(object_features_loaded)
                print(f"object_features_loaded shape is {object_features_loaded.shape}")
                combined_features = np.concatenate((combined_features, object_features_loaded), axis=0)

            # pad the combined_features to (5, 2048)
            print(f"len(combined_features): {len(combined_features)}")
            if len(combined_features) < 5:
                print("@@@@@@@@@@@@@@@@@@")
                combined_features = self.padding_feature(combined_features)
            print(f"combined_features shape is {combined_features.shape}")
            # save the combined_features as a pandaFrame file
            combined_features = pd.DataFrame(combined_features)
            print(combined_features.head())
            combined_features.to_csv("Extracted_HOFeatures_test/combined_features/" + file.split(".")[0] + ".csv",
                                     index=False)

            # # save the combined_features in txt file
            # with open("Extracted_HOFeatures/combined_features/" + file.split(".")[0] + ".txt", 'w') as f:
            #     for item in combined_features:
            #         f.write("%s\n" % item)
            # print(f"combined_features: {combined_features.shape}")
            # # print(f"combined_features: {combined_features}")
        pass


# Example usage
# change the working dir to msc2024_mengze
os.chdir("/scratch/users/lu/msc2024_mengze")
# extractor = GazeFeaturesExtractor(frames_dir="/scratch/users/lu/msc2024_mengze/Frames3/train_split1",test_dir="OP03-R01-PastaSalad-155200-157750-F003718-F003792")
extractor = GazeFeaturesExtractor(frames_dir="/scratch/users/lu/msc2024_mengze/Frames3/train_split1", gazepath_tosave="/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures_train/gaze")
extractor.process_frames(save_cropped_frames=False)

# extractor.concatenate_features()
