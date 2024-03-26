import os
import xml.etree.ElementTree as ET

Train_dir = "data_detection/Train"
Valid_dir = "data_detection/Valid"


def is_valid_bbox(bbox):
    x_min = float(bbox.find("xmin").text)
    y_min = float(bbox.find("ymin").text)
    x_max = float(bbox.find("xmax").text)
    y_max = float(bbox.find("ymax").text)
    return x_max > x_min and y_max > y_min


def find_invalid_bboxes(root_folder):
    for xml_file in os.listdir(root_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(root_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                if not is_valid_bbox(bbox):
                    print(f"Invalid bounding box found in file: {xml_path}")
                    print(f"Bounding box coordinates: [{bbox.find('xmin').text}, {bbox.find('ymin').text}, {bbox.find('xmax').text},"
                          f" {bbox.find('ymax').text}]")
                    image_file = xml_file[:-4] + ".jpg"
                    os.remove(os.path.join(Train_dir, image_file))
                    os.remove(os.path.join(Train_dir, xml_file))


find_invalid_bboxes(Train_dir)

for xml_file in os.listdir(Valid_dir):
    if xml_file.endswith(".xml"):
        tree = ET.parse(os.path.join(Valid_dir, xml_file))
        root = tree.getroot()
        if len(root.findall("object")) == 0:
            print(f"No bounding boxes exist for image: {xml_file[:-4]}")
            img_file = xml_file[:-4] + ".jpg"
            os.remove(os.path.join(Valid_dir, img_file))
            os.remove(os.path.join(Valid_dir,xml_file))
for xml_file in os.listdir(Train_dir):
    if xml_file.endswith(".xml"):
        tree = ET.parse(os.path.join(Train_dir, xml_file))
        root = tree.getroot()
        if len(root.findall("object")) == 0:
            print(f"No bounding boxes exist for image: {xml_file[:-4]}")
            img_file = xml_file[:-4] + ".jpg"
            os.remove(os.path.join(Train_dir, img_file))
            os.remove(os.path.join(Train_dir, xml_file))


