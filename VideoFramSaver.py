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
        """
        Get the list of video clips and videos in the given folder
        Output: list of video clips and list of videos
        """
        video_files = []
        videos = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.video_extensions):
                    relative_path = os.path.relpath(root, self.input_folder)
                    if not relative_path in videos:
                        videos.append(relative_path)
                    video_files.append((os.path.join(root, file), relative_path))
        return video_files, videos

    def save_frame(self, video_path, relative_path, video_name):
        # # if we want to save the frames in a separate folder for each video
        output_folder = os.path.join(self.output_base_folder, relative_path, video_name)
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
        start_frame = int(path_parts[-2].replace("F", ""))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // self.frame_samples)

        frame_count = 0
        spinner = itertools.cycle(['-', '/', '|', '\\'])

        for i in range(0, total_frames, interval + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                print("\nReached end of video or error reading a frame. Exiting.")
                break
            saved_frame_number = start_frame + i
            # output_path = os.path.join(output_folder, video_name, f"{video_name}_{saved_frame_count}.jpg")
            # save the frame with the frame number
            output_path = os.path.join(output_folder,f"{video_name}_{saved_frame_number}.jpg")
            cv2.imwrite(output_path, frame)

            elapsed_time = time.time() - start_time
            elapsed_time = elapsed_time / 60
            sys.stdout.write(f"\rProcessed {frame_count + 1}/{total_frames} frames {next(spinner)} --- Elapsed time: {elapsed_time:.2f} minis.")
            sys.stdout.flush()
            # time.sleep(0.1)

            frame_count += 1
            if frame_count == self.frame_samples:
                print("\nReached the maximum number of frames to be saved.")
                break
        cap.release()

    def process_videos(self):
        video_files, videos = self.get_video_files(self.input_folder)
        print(f"The clips to be processed: {len(video_files)}")
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
    output_base_folder = "Frames"
    new_directory = "/scratch/users/lu/msc2024_mengze"
    os.chdir(new_directory)
    frame_saver = VideoFrameSaver(input_folder, output_base_folder)
    start_time = time.time()
    frame_saver.process_videos()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
