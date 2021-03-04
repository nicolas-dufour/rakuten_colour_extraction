import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy, F1

class ResNet18(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 19)
        self.sigmoid = nn.Sigmoid()
        self.criterium = nn.BCEWithLogitsLoss()

        self.acc_train = Accuracy()
        self.f1_train = F1(num_classes=19, average='weighted')

        self.acc_val = Accuracy()
        self.f1_val = F1(num_classes=19, average='weighted')

        self.acc_test = Accuracy()
        self.f1_test = F1(num_classes=19, average='weighted')

    def forward(self, x):
        embedding = self.backbone(x)
        return self.sigmoid(embedding)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        labels = self.backbone(images)
        loss = self.criterium(labels, targets)
        self.log('train_loss', loss, on_epoch=True,on_step=True)
        self.acc_train(torch.sigmoid(labels), targets.long())
        self.f1_train(torch.sigmoid(labels), targets.long())
        return loss

    def training_epoch_end(self, loss):
        self.log('train_acc', self.acc_train.compute())
        self.log('train_f1_score', self.f1_train.compute())
        self.acc_train.reset()
        self.f1_train.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        labels = self.backbone(images)
        loss = self.criterium(labels, targets)
        self.log('valid_loss', loss, on_epoch=True)
        self.acc_val(torch.sigmoid(labels), targets.long())
        self.f1_val(torch.sigmoid(labels), targets.long())
    
    def validation_epoch_end(self, loss):
        self.log('val_acc', self.acc_val.compute())
        self.log('val_f1_score', self.f1_val.compute())
        self.acc_val.reset()
        self.f1_val.reset()
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        labels = self.backbone(images)
        self.acc_test(torch.sigmoid(labels), targets.long())
        self.f1_test(torch.sigmoid(labels), targets.long())
    
    def test_epoch_end(self, loss):
        self.log('test_acc', self.acc_test.compute())
        self.log('test_f1_score', self.f1_test.compute())
        self.acc_test.reset()
        self.f1_test.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)