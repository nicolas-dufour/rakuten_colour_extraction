from transformers import BertModel
import torch
from tqdm import tqdm

class Bert_classifier(torch.nn.Module):
    def __init__(self, nb_colors):
        super(Bert_classifier, self).__init__()
        self.input_layer = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
        self.drop_out = torch.nn.Dropout(0.3)
        self.dense = torch.nn.Linear(768, nb_colors)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.input_layer(ids, attention_mask = mask, token_type_ids = token_type_ids).pooler_output
        output_2 = self.drop_out(output_1)
        output = self.dense(output_2)
        return output

def train(nb_epochs, train_loader, device, model, optimizer):
    model.train()
    for e in range(nb_epochs):
        print(f'Number of epochs: {e}')
        for _, data in enumerate(tqdm(train_loader)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
            if _%5000==0:
                print(f'Epoch: {e}, Loss:  {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()