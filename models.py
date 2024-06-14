from transformers import AutoImageProcessor, AutoModelForObjectDetection
import lightning as L


class AtomicModelWrapper(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            config["model_parameters"]["model_name"]
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            config["model_parameters"]["model_name"]
        )

        self.config = config

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

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.step(batch, batch_idx)
        self.log('training_loss', loss)
        for k, v in loss_dict.items():
            self.log('train_' + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.step(batch, batch_idx)
        self.log('validation_loss', loss)
        for k, v in loss_dict.items():
            self.log('validation_' + k, v.item())
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
                "lr": self.config['model_parameters']['lr'],
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.config['model_parameters']['lr'],
            weight_decay=self.weight_decay,
            fused=True
        )

        return optimizer

    def get_processor(self):
        return self.processor