import os
from pathlib import Path

# Example of using os.path for handling Windows paths
file_name = "example.txt"
folder_path = "C:\\Users\\User\\Documents"

# Join path correctly
correct_path = os.path.join(folder_path, file_name)
print("Using os.path:", correct_path)

# Example using pathlib for more robust path handling
folder = Path("C:/Users/User/Documents")
file = folder / "example.txt"
print("Using pathlib:", file)

# Checking if the path exists
if file.exists():
    print("Path exists")
else:
    print("Path does not exist")
