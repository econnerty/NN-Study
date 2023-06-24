import os
import shutil

def copy_folder_names(source_dir, destination_dir):
    # Get a list of folder names in the source directory
    folder_names = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]
    
    # Create corresponding folders in the destination directory
    for folder_name in folder_names:
        destination_folder = os.path.join(destination_dir, folder_name)
        os.makedirs(destination_folder, exist_ok=True)
    
    print("Folder names copied successfully.")

# Specify the source and destination directories
source_directory = "./Data/Fruit/"
destination_directory = "./Data/FruitTest"

# Call the function to copy folder names
copy_folder_names(source_directory, destination_directory)
