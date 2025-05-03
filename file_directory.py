import os
from tqdm import tqdm

# Define base directory
base_dir = "D:/dataset_d/mscoco_yolo"

# Subdirectories containing images
subdirs = ["images/train2017", "images/val2017", "images/test2017"]

for subdir in subdirs:
    full_path = os.path.join(base_dir, subdir)
    output_file = os.path.join(base_dir, f"{subdir.replace('images/', '')}_paths.txt")  # Naming files like train2017_paths.txt
    
    relative_paths = []
    
    if os.path.exists(full_path):
        for filename in tqdm(os.listdir(full_path)):
            if filename.endswith((".jpg", ".png", ".jpeg")):  # Adjust file types if needed
                relative_paths.append(f"./{subdir}/{filename}")  # Adding `./` for relative paths

    # Save paths to corresponding file
    with open(output_file, "w") as f:
        f.write("\n".join(relative_paths))

    print(f"Saved {len(relative_paths)} image paths to {output_file}")
