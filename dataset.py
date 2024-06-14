import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd

import convert_to_coco as ctc


class AtomicDataset(torchvision.datasets.CocoDetection):
    def __init__(
            self,
            config: dict,
            processor,
            train: bool = True
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

    def __getitem__(self, idx):
        img, target = super(AtomicDataset, self).__getitem__(idx)
        image_id = self.ids[idx]

        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

    @staticmethod
    def create_train_test_split(config: dict) -> None:
        metadata = pd.read_csv(config['metadata_parameters']['path_to_unsplitted_metadata']).dropna().sample(frac=1)
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
    def collate_fn(batch, feature_extractor):
        pixel_values = [item[0] for item in batch]
        encoding = feature_extractor.pad_and_create_pixel_mask(
            pixel_values,
            return_tensors='pt'
        )
        labels = [item[1] for item in batch]
        batch = {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }
        return batch
