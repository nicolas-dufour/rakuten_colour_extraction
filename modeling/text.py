from transformers import BertModel
import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.classification import Accuracy, F1

class Bert_classifier(torch.nn.Module):
    def __init__(self, nb_colors):
        super().__init__()
        "self.input_layer = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')"
        self.input_layer = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-cha')
        self.drop_out = torch.nn.Dropout(0.3)
        self.dense = torch.nn.Linear(768, nb_colors)

    def forward(self, ids, mask):
        output_1 = self.input_layer(ids, attention_mask = mask, token_type_ids = None).pooler_output
        output_2 = self.drop_out(output_1)
        output = self.dense(output_2)
        return output

def train(nb_epochs, train_loader, val_loader, device, model, optimizer, model_path):
    model.train()
    val_loss, train_loss = [], []
    for e in range(nb_epochs):
        print(f'Number of epochs: {e}')
        current_loss = []
        for i, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask)
            loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
            current_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        validation_loss = evaluate(val_loader, model, device)
        print(f'Epoch: {e}, Training Loss:  {loss.item()}')
        print(f'Epoch: {e}, Validation Loss:  {validation_loss.item()}')
        val_loss.append(validation_loss.item())    
        train_loss.append(np.mean(current_loss))
    torch.save(model, model_path)
    return val_loss, train_loss


def evaluate(val_loader, model, device):
    losses = []
    with torch.no_grad(): 
        for data in val_loader:
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask)
            losses.append(torch.nn.BCEWithLogitsLoss()(outputs, targets).item())
    return np.mean(losses)



class Bert_classifier(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.input_layer = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
        self.drop_out = torch.nn.Dropout(0.3)
        self.dense = torch.nn.Linear(768, nb_colors)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.acc_train = Accuracy()
        self.f1_train = F1(num_classes=19, average='weighted')

        self.acc_val = Accuracy()
        self.f1_val = F1(num_classes=19, average='weighted')

        self.acc_test = Accuracy()
        self.f1_test = F1(num_classes=19, average='weighted')

    def forward(self, ids, mask):
        output_1 = self.input_layer(ids, attention_mask = mask, token_type_ids = None).pooler_output
        output_2 = self.drop_out(output_1)
        output = self.dense(output_2)
        return output

    def training_step(self, batch, batch_idx):
        ids, mask, targets = batch
        labels = self.forward(ids, mask)
        loss = self.criterion(labels, targets)
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
        ids, mask, targets = batch
        labels = self.forward(ids, mask)
        loss = self.criterion(labels, targets)
        self.log('valid_loss', loss, on_epoch=True)
        self.acc_val(torch.sigmoid(labels), targets.long())
        self.f1_val(torch.sigmoid(labels), targets.long())
    
    def validation_epoch_end(self, loss):
        self.log('val_acc', self.acc_val.compute())
        self.log('val_f1_score', self.f1_val.compute())
        self.acc_val.reset()
        self.f1_val.reset()
    
    def test_step(self, batch, batch_idx):
        ids, mask, targets = batch
        labels = self.forward(ids, mask)
        self.acc_test(torch.sigmoid(labels), targets.long())
        self.f1_test(torch.sigmoid(labels), targets.long())
    
    def test_epoch_end(self, loss):
        self.log('test_acc', self.acc_test.compute())
        self.log('test_f1_score', self.f1_test.compute())
        self.acc_test.reset()
        self.f1_test.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)