# --------------------------------------------------------------------------------
# run.py is the file to run the hand object detector module and other modules easily in remote server.
# --------------------------------------------------------------------------------


import subprocess
import os

from VideoFramSaver import VideoFrameSaver


def main():

    # define the path of the hand object detector module manually
    hand_object_detector = "/scratch/lu/msc2024_mengze/hand_object_detector"
    demo = "/scratch/lu/msc2024_mengze/hand_object_detector/demo.py"
    HOExtractor = "/scratch/lu/msc2024_mengze/hand_object_detector/HOExtractor.py"
    cuda = True

    # run module by user input
    # module = input("Enter the module name to run(demo/test): ")

    # module = "demo"
    module = "HOExtractor"

    if module == "demo":
        subprocess.run(["python", demo, "--cuda" if cuda else ""], cwd=hand_object_detector)

    elif module == "test":
        subprocess.run(["python3", "test_net.py", "--cuda" if cuda else "", "--save_name=handobj_100K"],
                       cwd=hand_object_detector)
        
    elif module == "HOExtractor":
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '2'
        subprocess.run(["python3", HOExtractor, "--cuda" if cuda else "", "--image_dir=/scratch/lu/msc2024_mengze/Frames_224/train_split1"], cwd=hand_object_detector)

    else:
        print("Invalid module name")

def extrct_frames():
    input_folder = "gaze_dataset/cropped_clips"
    output_base_folder = "Frames"
    new_directory = "/scratch/users/lu/msc2024_mengze"
    os.chdir(new_directory)
    frame_saver = VideoFrameSaver(input_folder, output_base_folder)
    frame_saver.process_videos()

if __name__ == "__main__":

    # select the module to run 1 for main, 2 for extracting_frames, 3 for both
    module = 1
    if module == 1:
        main()
    elif module == 2:
        extrct_frames()
        print("Frames extracted!")
    elif module == 3:
        extrct_frames()
        print("Frames extracted!")
        main()

