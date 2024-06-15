from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoFeatureExtractor
import lightning as L
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision


class AtomicModelWrapper(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        self.processor = AutoFeatureExtractor.from_pretrained(
            config["model_parameters"]["model_name"]
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            config["model_parameters"]["model_name"]
        )

        self.config = config

        torch.set_float32_matmul_precision('medium')
        self.metric = MeanAveragePrecision()

    def forward(self, pixel_values, pixel_mask):
        return self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask
        )

    def step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        labels = [
            {k: v.to(self.device) for k, v in t.items()}
            for t in batch['labels']
        ]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict, outputs

    def training_step(self, batch, batch_idx):
        loss, loss_dict, out = self.step(batch, batch_idx)
        self.log('training_loss', loss, prog_bar=True)

        for k, v in loss_dict.items():
            self.log('train_' + k, v.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, out = self.step(batch, batch_idx)

        self.log('validation_loss', loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log('validation_' + k, v.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                'params': [p for n, p in self.named_parameters()
                           if 'backbone' not in n and p.requires_grad]
            },
            {
                'params': [p for n, p in self.named_parameters()
                           if 'backbone' in n and p.requires_grad],
                "lr": self.config['optimizer_parameters']['learning_rate'],
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.config['optimizer_parameters']['learning_rate'],
            weight_decay=self.config['optimizer_parameters']['weight_decay'],
            fused=True
        )

        return optimizer

    def get_processor(self):
        return self.processor


class AtomicFasterRCNN(L.LightningModule):
    def __init__(self, config, num_classes):
        super(AtomicFasterRCNN, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                        num_classes)
        self.config = config

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, prog_bar=True)
        return losses

    # def validation_step(self, batch, batch_idx):
    #     images, targets = batch
    #     loss_dict = self.model(images, targets)
    #     print(loss_dict)
    #     losses = sum(loss for loss in loss_dict.values())
    #     self.log('val_loss', losses)
    #     return losses

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['optimizer_parameters']['learning_rate'],
            weight_decay=self.config['optimizer_parameters']['weight_decay'],
            fused=True
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [lr_scheduler]