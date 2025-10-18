# File: split_data.py
import os
import shutil
import random

print("Starting the data splitting process...")

# --- Configuration ---
# The main folder containing the original dataset.
# The CUB dataset usually has an 'images' subfolder, so we point to that.
source_dir = os.path.join('CUB_200_2011', 'images')

# The new folders where the split data will be stored.
train_dir = 'train_data'
validation_dir = 'validation_data'

# The ratio for the split (e.g., 0.8 means 80% for training, 20% for validation).
split_ratio = 0.8
# ---------------------

# Create the new train and validation directories if they don't exist
print(f"Creating directories: '{train_dir}' and '{validation_dir}'")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Get a list of all the class (bird species) folders
try:
    class_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    if not class_folders:
        print(f"Error: No subdirectories found in '{source_dir}'. Please check the path.")
        exit()
    print(f"Found {len(class_folders)} class folders.")
except FileNotFoundError:
    print(f"Error: The source directory '{source_dir}' was not found. Please make sure the path is correct.")
    exit()

# Loop through each class folder
for class_folder in class_folders:
    print(f"\nProcessing class: {class_folder}...")
    
    # Create corresponding subdirectories in train and validation folders
    os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_folder), exist_ok=True)
    
    # Get a list of all image files in the current class folder
    source_path = os.path.join(source_dir, class_folder)
    images = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    
    # Shuffle the list of images randomly
    random.shuffle(images)
    
    # Determine the split point
    split_point = int(len(images) * split_ratio)
    
    # Get the sublists of images for training and validation
    train_images = images[:split_point]
    validation_images = images[split_point:]
    
    print(f"  - Splitting {len(images)} images: {len(train_images)} for training, {len(validation_images)} for validation.")
    
    # Copy training images
    for image in train_images:
        shutil.copy(os.path.join(source_path, image), os.path.join(train_dir, class_folder, image))
        
    # Copy validation images
    for image in validation_images:
        shutil.copy(os.path.join(source_path, image), os.path.join(validation_dir, class_folder, image))

print("\n" + "="*50)
print("Data splitting complete! 🚀")
print(f"You now have '{train_dir}' and '{validation_dir}' folders ready for training.")
print("="*50)