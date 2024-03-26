import cv2.gapi
import torch
import cv2
import argparse
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Test_Image")
    parser.add_argument("--image_path", "--i", type=str, default="Test_Image.jpg")
    parser.add_argument("--checkpoint_path", "--c", type=str, default="Train Model/best.pt")
    parser.add_argument("--conf_threshold", "--t", type=float, default=0.5)
    args = parser.parse_args()
    return args


def test(args):
    classes = ["bus", "car", "motorbike", "truck", "microbus", "pickup-van"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float()[None, :, :, :].to(device)
    prediction = model(image)
    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"]
    scores = prediction[0]["scores"]
    for box, label, score in zip(boxes, labels, scores):
        if score > args.conf_threshold:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(ori_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 12)
            cv2.putText(ori_image, classes[label], (int(x_min), int(y_min)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128), 4, cv2.LINE_AA)
    cv2.imwrite("Test_image.jpg", ori_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    args = get_args()
    test(args)