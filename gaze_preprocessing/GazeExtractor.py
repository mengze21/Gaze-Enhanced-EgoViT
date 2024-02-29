import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog


def choose_file():
    """
    choose file from system filedialog and return the file path
    """
    # Choose the file
    # Create a root window but don't display it
    root = tk.Tk()
    root.withdraw()
    # Open the file dialog and get the selected file path
    file_path = filedialog.askopenfilename()

    return file_path


def auto_choose_txt_files(folder_path):
    """
    Automatically select all .txt files from the gaze_data folder and return their paths.
    """
    # List all .txt files in the specified directory
    gaze_data_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Check if the list of .txt files is empty
    if not gaze_data_files:
        raise FileNotFoundError("No .txt files found in the gaze_data directory.")
    
    return gaze_data_files


def save_data(data, fname=None):
    """
    Save the extract gaze data
    Args:
        data: dataFrame to save
        fname: when None default name gaze_data.txt
    """
    if fname is None:
        fname = 'gaze_data.txt'
    else:
        fname = os.path.basename(fname)
    # save the gaze data as a .txt file
    data.to_csv(fname, sep='\t', index=False)


def _str2frame(frame_str, fps=24):
    """
    change the frame format from 00:00:00:00 to integer
    Input:
        fps is default 24
    """
    splited_time = frame_str.split(':')
    assert len(splited_time) == 4

    time_sec = 3600 * int(splited_time[0]) \
               + 60 * int(splited_time[1]) + int(splited_time[2])
    frame_num = time_sec * fps + int(splited_time[3])

    return frame_num


def norm_filter_gaze(x, y, resolution, status_value):
    """
    Post-processing:

    1.Filter out of bound gaze points

    2.Normalize gaze into the range of 0-1

    If gaze point outside gaze_resolution, change the gaze status from 'Fixation' to 'Truncated'
    Args:
        x: gaze_x (width)
        y: gaze_y (height)
        resolution: gaze_resolution_x
        status_value: Status of gaze
    Returns: normalized gaze_x, gaze_y and gaze status
    """
    px = x
    py = y

    # truncate the gaze points
    if (px < 0 or px > (resolution[1] - 1) or
            py < 0 or py > (resolution[0]-1)):
        status_value = 'Truncated'

    px = min(max(0, px), resolution[1] - 1)
    py = min(max(0, py), resolution[0] - 1)

    # normalize the gaze
    px = px / resolution[1]
    py = py / resolution[0]
    return px, py, status_value


def parse_gaze(file_path, gaze_resolution=None):
    """
    Return the data as a pandas dataFrame

    The following information are kept: gaze_x (width), gaze_y (height), frame, gaze_type fixation
    """
    if gaze_resolution is None:
        # gaze resolution (default 1280*960)
        gaze_resolution = np.array([960, 1280], dtype=np.float32)

    # extract the gaze data from .txt file as pd dataframe
    gaze_data = pd.read_csv(file_path, delimiter='\t', comment='#')
    # make sure column name don't have space
    gaze_data.columns = gaze_data.columns.str.replace(' ', '')
    #print(gaze_data.head)
    # check gaze info version
    # BeGaze version 3.1 has 8 columns, BeGaze version 3.4 has 27 columns
    # For BeGaze version 3.1, columns 4-7 record gaze_x (width), gaze_y (height)
    # For BeGaze version 3.4, columns 6-7 record gaze_x (width), gaze_y (height)
    version = len(gaze_data.columns)
    if version == 8:
        gaze_x = gaze_data.columns[3]
        gaze_y = gaze_data.columns[4]
    elif version == 27:
        gaze_x = gaze_data.columns[5]
        gaze_y = gaze_data.columns[6]
        # change the data format for column 'Frame'
        gaze_data['Frame'] = gaze_data['Frame'].apply(_str2frame)

    # rename the columns
    gaze_data = gaze_data.rename(columns={gaze_x: 'xCoord', gaze_y: 'yCoord'})

    # keep needed column
    # get name of last column
    # (note: there are two names 'L Event Info' and 'B Evnet Info')
    last_column_name = gaze_data.columns[-1]
    gaze_data = gaze_data[['xCoord', 'yCoord', 'Frame', last_column_name]]

    # post-processing
    # apply normalization and filter function on columns 'xCoord' 'yCoord'
    # if gaze point outside gaze_resolution, change the gaze status from 'Fixation' to 'Truncated'
    gaze_data[['xCoord', 'yCoord', last_column_name]] = gaze_data.apply(
        lambda row: norm_filter_gaze(row['xCoord'], row['yCoord'], gaze_resolution, row[last_column_name]),
        axis=1, result_type='expand')

    # select the gaze info only with 'Fixation'
    gaze_data = gaze_data[gaze_data[last_column_name] == 'Fixation']

    return gaze_data


def extract_and_save_files(input_folder, output_folder):
    gaze_files = auto_choose_txt_files(input_folder)
    if not gaze_files:
        print("No .txt files found.")
        return

    # Create the target folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    for file_path in gaze_files:
        print(f"processing file: {file_path}")
        extracted_gaze = parse_gaze(file_path)

        # Creating a path for a new file
        base_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_folder, base_name.replace('.txt', '_processed.txt'))

        # save the gaze data as a .txt file
        extracted_gaze.to_csv(output_file_path, sep='\t', index=False)


def check_processed_file(folder1, folder2):
    """
    Compare the file counts between two folders.

    Args:
        folder1 (str): The path to the first folder.
        folder2 (str): The path to the second folder.

    Returns:
        None

    Example usage:
        compare_file_counts('/path/to/folder1', '/path/to/folder2')
    """
    # List all files in the folders
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Get counts of the files in the folders
    count1 = len(files1)
    count2 = len(files2)

    if count1 == count2:
        print("all data are proceed")
    elif count1 > count2:
        num = count1 - count2
        print(f"There are {num} files not processed")


def main():
    source_folder = "EGTA_Gaze+/gaze_data/gaze_data"
    target_folder = "project/data_processing/extracted_gaze_data"
    extract_and_save_files(source_folder, target_folder)
    check_processed_file(source_folder, target_folder)


if __name__ == "__main__":
    main()


