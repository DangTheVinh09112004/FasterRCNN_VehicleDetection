import argparse
import os
import shutil
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import Compose, ColorJitter, ToTensor, Normalize
from tqdm.autonotebook import tqdm

from VehicleDetection import VehicleDataset

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--epochs", "--e", type=int, default=50)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", "--b", type=int, default=8)
    parser.add_argument("--save_path", "--s", type=str, default="Train Model")
    parser.add_argument("--checkpoint_path", "--c", type=str, default="Train Model/last.pt")
    parser.add_argument("--tensorboard_path", "--t", type=str, default="Tensorboard")
    args = parser.parse_args()
    return args


def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device)
    train_transform = Compose([
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.465, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    valid_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.465, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    Train_data = VehicleDataset("data_detection/Train", transform=train_transform)
    Valid_data = VehicleDataset("data_detection/Valid", transform=valid_transform)
    Train_dataloader = DataLoader(
        dataset=Train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=collate_fn
    )
    Valid_dataloader = DataLoader(
        dataset=Valid_data,
        batch_size=args.batch_size,
        num_workers=6,
        collate_fn=collate_fn
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
    writer = SummaryWriter("tensorboard")
    os.makedirs(args.save_path, exist_ok=True)
    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.makedirs(args.save_path, exist_ok=True)
    best_map = -1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(Train_dataloader, colour=np.random.choice(["cyan", "yellow", "magenta", "green"]))
        for i, (images, targets) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            targets = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)}
                       for target in targets]
            loss_components = model(images, targets)
            losses = sum(loss for loss in loss_components.values())
            progress_bar.set_description("Epoch {}/{}. Loss: {:.4f}".format(epoch + 1, args.epochs, losses))
            writer.add_scalar("Train/Loss", losses, epoch * len(Train_dataloader) + i)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        model.eval()
        metrics = MeanAveragePrecision(iou_type="bbox")
        progress_bar = tqdm(Valid_dataloader, colour="pink")
        for i, (images, targets) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                predictions = model(images)
            targets = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)}
                       for target in targets]
            metrics.update(predictions, targets)
        map = metrics.compute()
        writer.add_scalar("Val/mAP", map["map"], epoch)
        writer.add_scalar("Val/mAP50", map["map_50"], epoch)
        writer.add_scalar("Val/mAP75", map["map_75"], epoch)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "best_map": map["map"],
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))
        if map["map"] > best_map:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))
            best_map = map["map"]


if __name__ == '__main__':
    args = get_args()
    train(args)