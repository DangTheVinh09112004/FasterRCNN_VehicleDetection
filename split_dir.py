import os
import shutil

output_dir = "data_detection"
source_dir = "data/train"
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "Train"))
os.makedirs(os.path.join(output_dir, "Valid"))

jpg_files = [file for file in os.listdir("data/train") if file.endswith(".jpg")]
total_files = len(jpg_files)

num_files_part1 = int(total_files * 0.8)

jpg_files_part1 = jpg_files[: num_files_part1]

for jpg_file in jpg_files_part1:
    shutil.copy(os.path.join(source_dir, jpg_file), os.path.join("data_detection/Train", jpg_file))
    xml_file = jpg_file[:-4] + ".xml"
    shutil.copy(os.path.join(source_dir, xml_file), os.path.join("data_detection/Train", xml_file))
for jpg_file in jpg_files:
    if jpg_file not in jpg_files_part1:
        shutil.copy(os.path.join(source_dir, jpg_file), os.path.join("data_detection/Valid", jpg_file))
        xml_file = jpg_file[:-4] + ".xml"
        shutil.copy(os.path.join(source_dir, xml_file), os.path.join("data_detection/Valid", xml_file))