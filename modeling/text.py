
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel
import torch
from pytorch_lightning.metrics.classification import Accuracy, F1
from torch import nn

class Bert_classifier(pl.LightningModule):
    def __init__(self, nb_colors, lr):
        super().__init__()
        self.lr = lr
        self.input_layer = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
        self.drop_out = torch.nn.Dropout(0.3)
        self.dense = torch.nn.Linear(768, nb_colors)
        self.criterion = nn.BCEWithLogitsLoss()

        self.acc_train = Accuracy()
        self.f1_train = F1(num_classes=nb_colors, average='weighted')

        self.acc_val = Accuracy()
        self.f1_val = F1(num_classes=nb_colors, average='weighted')

        self.acc_test = Accuracy()
        self.f1_test = F1(num_classes=nb_colors, average='weighted')

    def forward(self, ids, mask):
        output_1 = self.input_layer(ids, attention_mask = mask, token_type_ids = None).pooler_output
        output_2 = self.drop_out(output_1)
        output = self.dense(output_2)
        return output

    def training_step(self, batch, batch_idx):
        (ids, mask, targets, _) = batch
        labels = self.forward(ids, mask)
        loss = nn.BCEWithLogitsLoss()(labels, targets)
        self.log('train_loss', loss, on_epoch=True,on_step=True)
        self.acc_train(torch.sigmoid(labels), targets)
        self.f1_train(torch.sigmoid(labels), targets)
        return loss

    def training_epoch_end(self, loss):
        self.log('train_acc', self.acc_train.compute())
        self.log('train_f1_score', self.f1_train.compute())
        self.acc_train.reset()
        self.f1_train.reset()

    def validation_step(self, batch, batch_idx):
        (ids, mask, targets, _) = batch
        targets = targets.long()
        labels = self.forward(ids, mask)
        loss = nn.BCEWithLogitsLoss()(targets, targets)
        self.log('valid_loss', loss, on_epoch=True)
        self.acc_val(torch.sigmoid(labels), targets)
        self.f1_val(torch.sigmoid(labels), targets)
    
    def validation_epoch_end(self, loss):
        self.log('val_acc', self.acc_val.compute())
        self.log('val_f1_score', self.f1_val.compute())
        self.acc_val.reset()
        self.f1_val.reset()
    
    def test_step(self, batch, batch_idx):
        (ids, mask, targets, _) = batch
        targets = targets.long()
        labels = self.forward(ids, mask)
        self.acc_test(torch.sigmoid(labels), targets)
        self.f1_test(torch.sigmoid(labels), targets)
    
    def test_epoch_end(self, loss):
        self.log('test_acc', self.acc_test.compute())
        self.log('test_f1_score', self.f1_test.compute())
        self.acc_test.reset()
        self.f1_test.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)