import shutil
import os

source_dir = "../../../cats_front/"
dest_dir = "../../../Images_front_resc/"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Copy files from 0.jpg to 100.jpg
for i in range(821):  # 0 to 100 inclusive
    src_file = os.path.join(source_dir, f"{i}.jpg")
    dest_file = os.path.join(dest_dir, f"{i}.jpg")

    if os.path.exists(src_file):  # Check if file exists before copying
        shutil.copy(src_file, dest_file)
        print(f"Copied: {src_file} -> {dest_file}")
    else:
        print(f"File not found: {src_file}")
