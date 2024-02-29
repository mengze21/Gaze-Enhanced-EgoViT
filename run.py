# --------------------------------------------------------------------------------
# run.py is the main file to run the hand object detector module and other modules
# --------------------------------------------------------------------------------


import subprocess


def main():

    # define the path of the hand object detector module manually
    hand_object_detector = "/scratch/users/lu/msc2024_mengze/hand_object_detector"
    demo = "/scratch/users/lu/msc2024_mengze/hand_object_detector/demo.py"
    HOExtractor = "/scratch/users/lu/msc2024_mengze/hand_object_detector/HOExtractor.py"
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
        subprocess.run(["python3", HOExtractor, "--cuda" if cuda else ""], cwd=hand_object_detector)

    else:
        print("Invalid module name")


if __name__ == "__main__":
    main()
