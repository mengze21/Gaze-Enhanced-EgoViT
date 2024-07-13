import sys
import time
import itertools
import cv2
import os
import pandas as pd
import pandas as pd


class VideoFrameSaver:
    def __init__(self, input_folder, output_base_folder, video_list='train1', frame_samples=32, frame_scale= 0.25, video_extensions=None):
        self.input_folder = input_folder
        self.output_base_folder = output_base_folder
        self.frame_samples = frame_samples
        self.video_list = video_list
        self.frame_scale = frame_scale
        self.list = ['train1', 'train2', 'train3', 'test1', 'test2', 'test3']
        if video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov']
        else:
            self.video_extensions = video_extensions

    # def load_video_list(self, video_list_file):
    #     """
    #     Load the list of videos from the given file
    #     Output: list of videos
    #     """
    #     videos = []
    #     with open(video_list_file, "r") as f:
    #         for line in f:
    #             video_path = line.strip()
    #             videos.append(video_path)
    #     return videos
    

    def get_video_files(self, folder_path, video_list_file):
        """
        Get the list of video clips and videos in the given folder
        and filter the videos based on the given video list
        Output: list of video clips and list of videos
        """
        video_files = []
        videos = []
        if video_list_file == 'train1':
            video_list_path = 'dataset/train_split1.txt'
        elif video_list_file == 'test1':
            video_list_path = 'dataset/test_split1.txt'
        else:
            print(f"Invalid video list file: {video_list_file}")
            return
        # first column is the video name
        # spilt the row with space and get the first column
        with open(video_list_path, "r") as f:
            for line in f:
                video = line.strip().split(' ', 1)[0] + '.mp4'
                videos.append(video)
        print(f"Total videos in the list: {len(videos)}")
        print(videos[0])
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.video_extensions):
                    relative_path = os.path.relpath(root, self.input_folder)
                    if file in videos:
                        video_files.append((os.path.join(root, file), file))
        # print(f"Total video clips to be processed: {len(video_files)}")
        # print(video_files[0])
        return video_files, videos

    # def get_video_files(self, folder_path):
    #     """
    #     Get the list of video clips and videos in the given folder
    #     Output: list of video clips and list of videos
    #     """
    #     video_files = []
    #     videos = []
    #     for root, dirs, files in os.walk(folder_path):
    #         for file in files:
    #             if any(file.endswith(ext) for ext in self.video_extensions):
    #                 relative_path = os.path.relpath(root, self.input_folder)
    #                 if relative_path not in videos:
    #                     videos.append(relative_path)
    #                 video_files.append((os.path.join(root, file), relative_path))
    #     return video_files, videos

    def save_frame(self, video_path, relative_path, video_name, frame_count=32):
        # # if we want to save the frames in a separate folder for each video
        # output_folder = os.path.join(self.output_base_folder, relative_path, video_name)
        output_folder = os.path.join(self.output_base_folder, video_name)
        # output_folder = self.output_base_folder
        # print(f"Output folder: {output_folder}")
        # print(f"Video name: {video_name}")
        # print(f"Video path: {video_path}")
        # print(f"Relative path: {relative_path}")
        # Check if the output folder exists; if not, create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # parse the path parts to get start frame number of the video
        path_parts = video_name.split("-")
        start_frame_num = int(path_parts[-2].replace("F", ""))
        # end_frame_num = int(path_parts[-1].replace("F", ""))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        spinner = itertools.cycle(['-', '/', '|', '\\'])

        # new algorithm for extract frame
        interval = max(1.0, (total_frames - 1) / (frame_count - 1))

        for i in range(frame_count):
            frame_id = round(i * interval)
            if i < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    print("\nReached end of video or error reading a frame. Exiting.")
                    break
                saved_frame_num = start_frame_num + frame_id
                output_path = os.path.join(output_folder, f"{video_name}_{saved_frame_num}.jpg")
            else:
                # print("\nReached the end of video using padding.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()
                if not ret:
                    print("\nError reading the last frame. Exiting.")
                    break
                saved_frame_num = start_frame_num + total_frames - 1
                output_path = os.path.join(output_folder, f"{video_name}_{saved_frame_num}_{i-total_frames}.jpg")

            # if i >= total_frames:
            #     print("\nReached the end of video")
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            #     ret, frame = cap.read()
            #     print()
            #     saved_frame_number = start_frame + total_frames - 1
            #     output_path = os.path.join(output_folder, f"{video_name}_{saved_frame_number}_{i+1}.jpg")
            # else:
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            #     ret, frame = cap.read()
            #     saved_frame_number = start_frame + i
            #     output_path = os.path.join(output_folder, f"{video_name}_{saved_frame_number}.jpg")
            # resize the frame to to 1/4 of the original size
            frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)

            cv2.imwrite(output_path, frame)

            elapsed_time = time.time() - start_time
            elapsed_time = elapsed_time / 60
            sys.stdout.write(
                f"\rProcessed {i + 1}/{total_frames} frames {next(spinner)} --- Elapsed time: {elapsed_time:.2f} minis.")
            sys.stdout.flush()
        cap.release()
        """
        # if total_frame / 32 < 1.5 then take all frame
        if total_frames <= 48:
            interval = 0
        else:
            # use round
            interval = max(1, round(total_frames / self.frame_samples))
            # rounds the result down to the nearest whole number
            # interval = max(1, total_frames // self.frame_samples)

        frame_count = 0
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        

        for i in range(0, total_frames + 1, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                print("\nReached end of video or error reading a frame. Exiting.")
                break
            saved_frame_number = start_frame + i
            # output_path = os.path.join(output_folder, video_name, f"{video_name}_{saved_frame_count}.jpg")
            # save the frame with the frame number
            output_path = os.path.join(output_folder, f"{video_name}_{saved_frame_number}.jpg")
            # cv2.imwrite(output_path, frame)

            elapsed_time = time.time() - start_time
            elapsed_time = elapsed_time / 60
            sys.stdout.write(
                f"\rProcessed {frame_count + 1}/{total_frames} frames {next(spinner)} --- Elapsed time: {elapsed_time:.2f} minis.")
            sys.stdout.flush()
            # time.sleep(0.1)

            frame_count += 1
            if frame_count == self.frame_samples:
                print("\nReached the maximum number of frames (32) to be saved.")
                break
   
        """
    def process_videos(self):
        video_files, videos = self.get_video_files(self.input_folder, self.video_list)
        # deos = self.get_video_files(self.input_folder, self.video_list)
        # print(f"The clips to be processed: {len(video_files)}")
        print(f"The videos to be processed: {len(videos)}")
        video_num = 0

        for video_path, relative_path in video_files:
            print("\n-----------------------------------------------")
            print(f"Processing {video_num + 1}/{len(video_files)} video.")
            # # use the following line to process given number of videos for testing
            # if video_num == 300:
            #     print(f"\nOnly process {video_num} videos for testing and exit.")
            #     break

            # print(f"Processing {video_num + 1}/{len(video_files)} video.")
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            # print(f"Video name: {video_name}")
            # print(f"Video path: {video_path}")
            # print(f"Relative path: {relative_path}")
            self.save_frame(video_path, relative_path, video_name)
            video_num += 1

        print("--------Done!--------")


if __name__ == "__main__":
    # input_folder = "gaze_video"
    input_folder = "gaze_dataset/cropped_clips"
    output_base_folder = "Frames3/test_split1"
    new_directory = "/scratch/users/lu/msc2024_mengze"
    os.chdir(new_directory)
    frame_saver = VideoFrameSaver(input_folder, output_base_folder, video_list='test1')
    start_time = time.time()
    frame_saver.process_videos()
    end_time = time.time()
    time_taken_minutes = (end_time - start_time) / 60
    print(f"Time taken: {time_taken_minutes:.2f}mins")
