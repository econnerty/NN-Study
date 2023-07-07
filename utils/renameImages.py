import os
import pandas as pd

def rename_files_in_dir(dir_path, extension=".jpg"):
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(extension)])
    
    # Get the name of the current folder
    folder_name = os.path.basename(os.path.normpath(dir_path))
    
    metadata = []
    for i, file in enumerate(files, start=1):
        new_name = f"{i:04}{extension}"
        os.rename(os.path.join(dir_path, file), os.path.join(dir_path, new_name))
        metadata.append({"file_name": new_name, "text": folder_name})

    # Create a DataFrame from the metadata and save it as a .csv file
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(dir_path, "metadata.csv"), index=False)

# Usage
rename_files_in_dir("../Data/Pencils/train/pencil")
