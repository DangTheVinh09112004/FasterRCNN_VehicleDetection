from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import torch
import cv2
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Test_Video")
    parser.add_argument("--video_path", "--v", type=str, default="Video.mp4")
    parser.add_argument("--out_path", "--o", type=str, default="Out_Video.mp4")
    parser.add_argument("--checkpoint_path", "--c", type=str, default="Train Model/best.pt")
    parser.add_argument("--conf_threshold", "--t", type=float, default=0.5)
    args = parser.parse_args()
    return args


def test(args):
    classes = ["bus", "car", "motorbike", "truck", "microbus", "pickup-van"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2().to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    cap = cv2.VideoCapture(args.video_path)
    out = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.
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
            if score >= args.conf_threshold:
                x_min, y_min, x_max, y_max = map(int, box)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                scaling_factor = 0.03
                font_scale = min(bbox_width, bbox_height) * scaling_factor
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, classes[label].format(score), (x_min, y_min),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (128, 0, 128), 2, cv2.LINE_AA)
        out.write(frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    args = get_args()
    test(args)