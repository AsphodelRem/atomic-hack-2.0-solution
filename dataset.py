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


class AtomicDataset(torchvision.datasets.CocoDetection):
    def __init__(
            self,
            config: dict,
            processor,
            train: bool = False
    ):
        metadata = config['metadata_parameters']['path_to_train_metadata'] \
            if train else config['metadata_parameters']['path_to_test_metadata']

        super(AtomicDataset, self).__init__(
            config['metadata_parameters']['path_to_data'],
            metadata
        )

        self.processor = processor
        self.config = config
        self.train = train
        file = open(config['metadata_parameters']['path_to_train_metadata'])
        self.metadata = json.load(file)

    def __getitem__(self, idx):
        target = self.metadata['annotations'][idx]
        image_id = target['image_id']

        img = Image.open(
            os.path.join(
                self.config['metadata_parameters']['path_to_data'],
                self.metadata['images'][image_id + 1]["file_name"]
            )
        ).convert('RGB')

        if self.train:
            img = augment(np.array(img))

        target = {'image_id': image_id, 'annotations': [target]}

        encoding = self.processor(images=img, annotations=target, return_tensors="pt")

        pixel_values = encoding['pixel_values'].squeeze()
        target = encoding['labels'][0]

        return pixel_values, target

    @staticmethod
    def create_train_test_split(config: dict) -> None:
        metadata = pd.read_csv(config['metadata_parameters']['path_to_unsplitted_metadata']).dropna()

        split_index = int(len(metadata) * config['metadata_parameters']['split_ratio'])

        # Split on train and test
        train_metadata = metadata[:split_index]
        test_metadata = metadata[split_index:]

        # Save as json in COCO format
        ctc.convert_csv_to_coco(
            train_metadata,
            config['metadata_parameters']['path_to_train_metadata']
        )
        ctc.convert_csv_to_coco(
            test_metadata,
            config['metadata_parameters']['path_to_test_metadata']
        )

    @staticmethod
    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }
        return batch


class AtomicDatasetV2(Dataset):
    def __init__(self, csv_file, root_dir, config, transforms=None):
        self.annotations = pd.read_csv(csv_file).dropna()
        self.root_dir = root_dir
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
        xmin = rel_x * w
        ymin = rel_y * h
        xmax = xmin + width * w
        ymax = ymin + height * h

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        img = F.to_tensor(img)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def create_train_test_split(config: dict) -> None:
        metadata = pd.read_csv(config['metadata_parameters']['path_to_unsplitted_metadata']).dropna()

        split_index = int(len(metadata) * config['metadata_parameters']['split_ratio'])

        # Split on train and test
        train_metadata = metadata[:split_index]
        test_metadata = metadata[split_index:]

        train_metadata.to_csv(config['metadata_parameters']['path_to_train_metadata'])
        test_metadata.to_csv(config['metadata_parameters']['path_to_test_metadata'])
