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


class NFNet(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model('dm_nfnet_f0', pretrained=True)
        self.backbone.head.fc = nn.Linear(self.backbone.head.fc.in_features, 19)
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

class Deit(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.backbone.head_dist = nn.Identity()
        self.head = nn.Linear(in_features, 19)
        self.head_dist = nn.Linear(in_features, 19)
        self.sigmoid = nn.Sigmoid()
        self.criterium = nn.BCEWithLogitsLoss()

        self.acc_teacher_train = Accuracy()
        self.f1_teacher_train = F1(num_classes=19, average='weighted')

        self.acc_student_train = Accuracy()
        self.f1_student_train = F1(num_classes=19, average='weighted')

        self.acc_teacher_val = Accuracy()
        self.f1_teacher_val = F1(num_classes=19, average='weighted')

        self.acc_student_val = Accuracy()
        self.f1_student_val = F1(num_classes=19, average='weighted')

        self.acc_teacher_test = Accuracy()
        self.f1_teacher_test = F1(num_classes=19, average='weighted')

        self.acc_student_test = Accuracy()
        self.f1_student_test = F1(num_classes=19, average='weighted')

    def forward(self, x):
        if self.training:
            x, _ = self.backbone(x)
        else:
            x= self.backbone(x)
        main = self.head(main)
        dist = self.head_dist(main)
        return main, dist
    
    def get_probas(self, x):
        if self.training:
            x, _ = self.backbone(x)
        else:
            x= self.backbone(x)
        main = self.head(main)
        dist = self.head_dist(main)
        return nn.sigmoid(main), nn.sigmoid(dist)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        labels_teacher, labels_student = self(images)

        loss_teacher = self.criterium(labels_teacher, targets)
        self.acc_teacher_train(torch.sigmoid(labels_teacher), targets.long())
        self.f1_teacher_train(torch.sigmoid(labels_teacher), targets.long())

        labels_teacher_hard = (labels_teacher>0.5).long()
        loss_student = self.criterium(labels_student, labels_teacher_hard)

        self.acc_student_train(torch.sigmoid(labels_student), targets.long())
        self.f1_student_train(torch.sigmoid(labels_student), targets.long())

        loss = (loss_teacher + loss_student)/2

        self.log('train_loss', loss, on_epoch=True,on_step=True)
        return loss

    def training_epoch_end(self, loss):
        self.log('train_acc_teacher', self.acc_teacher_train.compute())
        self.log('train_f1_score_teacher', self.f1_teacher_train.compute())
        self.log('train_acc_student', self.acc_student_train.compute())
        self.log('train_f1_score_student', self.f1_student_train.compute())
        self.acc_teacher_train.reset()
        self.f1_teacher_train.reset()
        self.acc_student_train.reset()
        self.f1_student_train.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        labels_teacher, labels_student = self(images)

        loss_teacher = self.criterium(labels_teacher, targets)
        self.acc_teacher_val(torch.sigmoid(labels_teacher), targets.long())
        self.f1_teacher_val(torch.sigmoid(labels_teacher), targets.long())

        labels_teacher_hard = (labels_teacher>0.5).long()
        loss_student = self.criterium(labels_student, labels_teacher_hard)

        self.acc_student_val(torch.sigmoid(labels_student), targets.long())
        self.f1_student_val(torch.sigmoid(labels_student), targets.long())

        loss = (loss_teacher + loss_student)/2

        self.log('valid_loss', loss, on_epoch=True,on_step=True)
    
    def validation_epoch_end(self, loss):
        self.log('val_acc_teacher', self.acc_teacher_val.compute())
        self.log('val_f1_score_teacher', self.f1_teacher_val.compute())
        self.log('val_acc_student', self.acc_student_val.compute())
        self.log('val_f1_score_student', self.f1_student_val.compute())
        self.acc_teacher_val.reset()
        self.f1_teacher_val.reset()
        self.acc_student_val.reset()
        self.f1_student_val.reset()
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        labels_teacher, labels_student = self(images)
        self.acc_teacher_test(torch.sigmoid(labels_teacher), targets.long())
        self.f1_teacher_test(torch.sigmoid(labels_teacher), targets.long())
        self.acc_student_test(torch.sigmoid(labels_student), targets.long())
        self.f1_student_test(torch.sigmoid(labels_student), targets.long())
    
    def test_epoch_end(self, loss):
        self.log('test_acc_teacher', self.acc_teacher_test.compute())
        self.log('test_f1_score_teacher', self.f1_teacher_test.compute())
        self.log('test_acc_student', self.acc_student_test.compute())
        self.log('test_f1_score_student', self.f1_student_test.compute())
        self.acc_teacher_test.reset()
        self.f1_teacher_test.reset()
        self.acc_student_test.reset()
        self.f1_student_test.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)