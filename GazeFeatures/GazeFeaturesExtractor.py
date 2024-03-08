# ------------------------------------------
# 
# This file contains the implementation of the GazeFeaturesExtractor class.
# ------------------------------------------

import torch
import cv2


def build_bounding_box(x, y, W, H):
    # Calculate the coordinates of the top-left corner
    # Make sure the coordinates of bounding box are in the image
    top_left_x = max(0, int(x - (W / 2)))
    top_left_y = max(0, int(y + (H / 2)))

    # Calculate the coordinates of the bottom-right corner
    # Make sure the coordinates of bounding box are in the image
    bottom_right_x = min(640, int(x + (W / 2)))
    bottom_right_y = min(480, int(y - (H / 2)))

    # Return the bounding box coordinates
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def load_and_crop_frame(file_path, center_x, center_y, width, height):
    # Load the frame from file path
    frame = cv2.imread(file_path)
    print(frame)
    # Build the bounding box
    x1, y1, x2, y2 = build_bounding_box(center_x, center_y, width, height)
    print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
    print(f"frame.shape: {frame.shape}")
    # Crop the frame
    cropped_frame = frame[y2:y1, x1:x2]
    print(f"cropped_frame: {cropped_frame}")
    # if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
    #     raise ValueError("Error: The bounding box is out of the frame.")
    return cropped_frame


# save the cropped frame to a file
def save_cropped_frame(cropped_frame, output_file_path):
    # Save the cropped frame to a file
    cv2.imwrite(output_file_path, cropped_frame)


def transform_coordinate(x, y):
    """
    Transform a normalized coordinate to 640x480
    and change the (0, 0) position
    """
    x = x * 640
    y = y * 480
    y = 480 - y
    return int(x), int(y)


# Example usage
x_cor = 0.28453125
y_cor = 0.7057395833333333
width = 150
height = 150

# transform the coordinate from a coordinate origin at top left to bottom left
# y_cor = 480 - y_cor

# transform a coordinate to 640x480
x_cor, y_cor = transform_coordinate(x_cor, y_cor)

cropped_image = load_and_crop_frame(
    "/scratch/users/lu/msc2024_mengze/Frames/OP03-R01-PastaSalad/OP03-R01-PastaSalad-100720-102010-F002414-F002452/OP03-R01-PastaSalad-100720-102010-F002414-F002452_31.jpg",
    x_cor, y_cor, width, height)

save_cropped_frame(cropped_image, "GazeFeatures/cropped_frame.jpg")
print("Cropped frame saved successfully.")
# bounding_box = build_bounding_box(center_x, center_y, width, height)
# print(bounding_box)
