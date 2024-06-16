import lightning as L
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class AtomicFasterRCNN(L.LightningModule):
    def __init__(self, config, num_classes):
        super(AtomicFasterRCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes
        )
        self.config = config

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['optimizer_parameters']['learning_rate'],
            weight_decay=self.config['optimizer_parameters']['weight_decay'],
            fused=True
        )
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer
