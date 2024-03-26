import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ColorJitter, ToTensor, Normalize
from typing import Optional, Callable, Tuple, Dict


class VehicleDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        self.classes = ["bus", "car", "motorbike", "truck", "microbus", "pickup-van"]
        self.image_files = []
        self.annotation_file = []
        for file in os.listdir(root):
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                xml_file = file[:-4] + ".xml"
                xml_path = os.path.join(root, xml_file)
                self.image_files.append(image_path)
                self.annotation_file.append(xml_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, item) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path = self.image_files[item]
        annotation_path = self.annotation_file[item]
        images = self.transform(Image.open(image_path))
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            x_min = float(bbox.find("xmin").text)
            y_min = float(bbox.find("ymin").text)
            x_max = float(bbox.find("xmax").text)
            y_max = float(bbox.find("ymax").text)
            bboxes.append([x_min, y_min, x_max, y_max])
            label = obj.find("name").text
            labels.append(self.classes.index(label))
        targets = {
            "boxes": torch.FloatTensor(bboxes),
            "labels": torch.LongTensor(labels)
        }
        return images, targets


if __name__ == '__main__':
    data_transform = Compose([
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    dataset = VehicleDataset("data/train",transform=data_transform)
    image, target = dataset[2]
    print(image)
    print(target)