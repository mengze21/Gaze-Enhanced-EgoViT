import sys
import time
import itertools
import cv2
import os

class VideoFrameSaver:
    def __init__(self, input_folder, output_base_folder, frame_samples=32, video_extensions=None):
        self.input_folder = input_folder
        self.output_base_folder = output_base_folder
        self.frame_samples = frame_samples
        if video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov']
        else:
            self.video_extensions = video_extensions

    def get_video_files(self, folder_path):
        video_files = []
        # print(f"Current working directory: {os.getcwd()}")
        for root, dirs, files in os.walk(folder_path):
            # print(f"Searching in {root}")
            for file in files:
                if any(file.endswith(ext) for ext in self.video_extensions):
                    relative_path = os.path.relpath(root, self.input_folder)
                    video_files.append((os.path.join(root, file), relative_path))
        return video_files

    def save_frame(self, video_path, relative_path, video_name):
        # # if we want to save the frames in a separate folder for each video
        # output_folder = os.path.join(self.output_base_folder, relative_path, video_name)
        output_folder = self.output_base_folder
        print(f"Output folder: {output_folder}")
        print(f"Video name: {video_name}")
        print(f"Video path: {video_path}")
        print(f"Relative path: {relative_path}")
        # Check if the output folder exists; if not, create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // self.frame_samples)

        frame_count = 0
        saved_frame_count = 0
        spinner = itertools.cycle(['-', '/', '|', '\\'])

        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                print("\nReached end of video or error reading a frame. Exiting.")
                break

            output_path = os.path.join(output_folder, f"{video_name}_{saved_frame_count}.jpg")
            # print(f"outpu_path")
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1

            sys.stdout.write(f"\rProcessed {frame_count + 1}/{total_frames} frames {next(spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)

            frame_count += 1
            if frame_count == self.frame_samples:
                print("\nReached the maximum number of frames to be saved.")
                break
        cap.release()

    def process_videos(self):
        video_files = self.get_video_files(self.input_folder)
        print(f"The video to be processed: {len(video_files)}")
        video_num = 0

        for video_path, relative_path in video_files:
            # use the following line to process given number of videos for testing
            if video_num == 3:
                print("Only process 3 videos for testing and exit.")
                break
            print(f"Processing {video_num + 1}/{len(video_files)} video.")
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"Video name: {video_name}")
            print(f"Video path: {video_path}")
            print(f"Relative path: {relative_path}")
            self.save_frame(video_path, relative_path, video_name)
            video_num += 1

        print("Done!")


if __name__ == "__main__":
    # input_folder = "gaze_video"
    input_folder = "gaze_dataset/cropped_clips"
    output_base_folder = "Frames"
    new_directory = "/scratch/users/lu/msc2024_mengze"
    os.chdir(new_directory)
    frame_saver = VideoFrameSaver(input_folder, output_base_folder)
    frame_saver.process_videos()
