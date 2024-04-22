# expand the colunm of every line for train_split.txt
# make sure every line has same number of columns
# the expand value is Nan

def pad_and_save_txt(filepath, output_filepath):
    """
    Reads a text file, pads each row with empty strings to the maximum column count, and saves the modified data to a new text file.

    Args:
        filepath (str): The path to the input text file.
        output_filepath (str): The path to the output text file.
    """

    # Read the file line by line
    with open(filepath, 'r') as f, open(output_filepath, 'w') as out_f:
        lines = f.readlines()

        # Find the maximum number of columns
        max_columns = 0
        for line in lines:
            columns = len(line.strip().split())
            max_columns = max(max_columns, columns)

        # Pad and write each line
        for line in lines:
            elements = line.strip().split()
            padded_row = " ".join(elements + [""] * (max_columns - len(elements)))
            out_f.write(padded_row + "\n")

# Example usage
filepath = "dataset/test_split1.txt"  # Replace with your actual file path
output_filepath = "dataset/test_split12.txt"  # Replace with your desired output file path

pad_and_save_txt(filepath, output_filepath)
print("Done!")