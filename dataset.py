import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
from PIL import Image
from augmentations import augment
from torchvision.transforms import functional as F

import convert_to_coco as ctc


class AtomicDataset(Dataset):
    def __init__(self, config, is_train: bool=False, transforms=None):
        csv_file = config['metadata_parameters']['path_to_train_metadata'] if is_train \
            else config['metadata_parameters']['path_to_test_metadata']
        
        self.annotations = pd.read_csv(csv_file).dropna()
        self.root_dir = config['metadata_parameters']['path_to_data']
        self.transforms = transforms
        self.config = config

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])
        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []

        label = self.annotations.iloc[idx, 2]
        rel_x = self.annotations.iloc[idx, 3]
        rel_y = self.annotations.iloc[idx, 4]
        height = self.annotations.iloc[idx, 5]
        width = self.annotations.iloc[idx, 6]

        w, h = img.size
        normal_width = width * w
        normal_height = height * h
        xmin = rel_x * w - normal_width // 2
        ymin = rel_y * h - normal_height // 2
        xmax = xmin + normal_width
        ymax = ymin + normal_height

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        img = F.to_tensor(img)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def create_train_test_split(config: dict) -> None:
        metadata = pd.read_csv(
            config['metadata_parameters']['path_to_unsplitted_metadata']
        ).dropna()
        split_index = int(len(metadata) * config['metadata_parameters']['split_ratio'])

        # Split on train and test
        train_metadata = metadata[:split_index]
        test_metadata = metadata[split_index:]

        train_metadata.to_csv(
            config['metadata_parameters']['path_to_train_metadata']
        )
        test_metadata.to_csv(
            config['metadata_parameters']['path_to_test_metadata']
        )
