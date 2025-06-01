import os
from pathlib import Path
import re
import shutil


"""
Code to extract files by the depth of the buried UXO in cm for a given dataset.
Special case is free_zone
"""
def list_folders():
    base_path = Path('../data/DatainBrief_dataset_1')
    output_base = Path('../data/sorted')
    # Check if directory exists
    if not os.path.exists(base_path):
        print(f"Error: {base_path} directory does not exist")
        return
    
    # In each flight set
    for folder in os.listdir(base_path):
        date_path = Path(base_path, folder)
        # In each type of images
        if os.path.isdir(date_path):
            for photo_type_dir in os.listdir(date_path):
                if photo_type_dir in ["JPG", "R_JPG"]:
                    photo_type_dir_path = Path(date_path, photo_type_dir)
                    # In each set of images
                    if os.path.isdir(photo_type_dir_path):
                        for zone_dir in os.listdir(photo_type_dir_path):
                            zone_dir_path = Path(photo_type_dir_path, zone_dir)
                            # Find the distance from the zone name
                            zone_name_lower = zone_dir.lower()
                            if "free zone" in zone_name_lower:
                                output_dir = Path(output_base, "free_zone")
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)

                                # Copy image to output directory
                                for image_file in os.listdir(zone_dir_path):
                                    shutil.copy(Path(zone_dir_path, image_file), Path(output_dir, f"free_zone_{image_file}"))
                                continue

                            if "cm" in zone_name_lower:
                                # Find the position of 'cm'
                                cm_index = zone_name_lower.find("cm")
                                if cm_index != -1:
                                    # Extract substring to the left of 'cm'
                                    left_part = zone_name_lower[:cm_index][::-1]
                                    print(left_part[::-1])
                                    # Find all numbers after the first letter in left_part
                                    match = re.search(r'(\d+)', left_part)
                                    if match:
                                        ## Check distanc in cm to sort images
                                        distance = match.group(1)[::-1]
                                        print(distance)
                                        # Create output directory
                                        output_dir = Path(output_base, distance + "cm")
                                        if not os.path.exists(output_dir):
                                            os.makedirs(output_dir)

                                        # Copy image to output directory
                                        for image_file in os.listdir(zone_dir_path):
                                            shutil.copy(Path(zone_dir_path, image_file), Path(output_dir, image_file))

if __name__ == "__main__":
    list_folders()
