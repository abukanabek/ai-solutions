import torch 
import torchvision.models as models 
from torchvision.models import ResNet50_Weights 
import os 
os.environ["TORCH_HOME"] = "torchmodels"

"""
Baseline: finetune Faster R-CNN on annotated data and produce submission.csv.

Usage:
    uv run baseline/baseline.py --train-csv data/train.csv --test-csv data/test.csv
"""

import matplotlib.pyplot as plt
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np

import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, fcos_resnet50_fpn, ssd300_vgg16,
    ssdlite320_mobilenet_v3_large, fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_320_fpn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm


class DetectionDataset(Dataset):
    def __init__(self, rows: list[dict], data_dir: Path, cat_ids: list[int]):
        self.data_dir = data_dir
        self.cat_map = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        self.samples = []
        for r in rows:
            ann = json.loads(r["annotation"])
            if ann and "bbox" in ann[0]:
                self.samples.append((r, ann))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row, anns = self.samples[idx]
        img_path = self.data_dir / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = to_tensor(img)

        boxes = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_map[a["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(row["datapointID"])]),
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def solve(train_csv='data/train.csv', test_csv='data/test.csv', data_dir='data', output_csv='submission.csv', checkpoint_dir='checkpoints', epochs=30, batch_size=4, lr=0.005, num_workers=2, score_threshold=0.42):
    random.seed(42)
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_csv(Path(train_csv))

    cat_id_set = set()
    for r in train_rows:
        for det in json.loads(r["annotation"]):
            cat_id_set.add(det["category_id"])
    cat_ids = sorted(cat_id_set)
    num_classes = len(cat_ids) + 1
    label_to_cat_id = {i + 1: cid for i, cid in enumerate(cat_ids)}
    dataset = DetectionDataset(train_rows, data_dir, cat_ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_rows = read_csv(Path(test_csv))
    model.eval()

    submission_rows = []
    for row in tqdm(test_rows, desc="Predicting"):
        img_path = data_dir / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        tensor = to_tensor(img).unsqueeze(0).to(device)

        img_np = np.array(img)
        
        with torch.no_grad():
            outputs = model(tensor)

        output = outputs[0]
        boxes = output["boxes"].cpu()
        scores = output["scores"].cpu()
        labels = output["labels"].cpu()
        # print(img_path)
        # print(labels)
        # for box in boxes[7:]:
        #     cv2.rectangle(img_np,(int(box[0].item()),int(box[1].item())),(int(box[2].item()),int(box[3].item())),(0,255,0),2)
        #     break
        # plt.imshow(img_np)
        # break

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if label.item() not in label_to_cat_id:
                continue
            if score.item() < score_threshold:
                continue
            if random.random() < 0.4:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                    "category_id": label_to_cat_id[label.item()],
                    "score": round(score.item(), 4),
                }
            )

        submission_rows.append(
            {
                "subtaskID": 1,
                "datapointID": row["datapointID"],
                "answer": json.dumps(detections),
            }
        )
    # return
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subtaskID", "datapointID", "answer"])
        writer.writeheader()
        writer.writerows(submission_rows)

    print(f"Wrote {output_csv} ({len(submission_rows)} rows)")

solve()


