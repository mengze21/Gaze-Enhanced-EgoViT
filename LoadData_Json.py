import numpy as np
import json

# Read data back from the JSON file
with open('/Volumes/Mengze/OP01-R01-PastaSalad_000016.json', 'r') as f:
    data_loaded = json.load(f)

# Convert lists back to NumPy arrays
array1_loaded = np.array(data_loaded["object_features"])
array2_loaded = np.array(data_loaded["object_box"])

print(array2_loaded.dtype)
print(array2_loaded.shape)
print(array2_loaded[0, 0:4])
print(data_loaded.keys())
