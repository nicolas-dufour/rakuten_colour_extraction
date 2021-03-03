from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from data.labels import Labels
import pandas as pd
import numpy as np 
import torch

class Bert_dataset(Dataset):
    def __init__(self, data_path, X_path, y_path, max_len):
      self.labels_obj = Labels(data_path + y_path)
      self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
      self.max_len = max_len
      self.load_datasets(data_path, X_path, y_path)
      self.remove_empty_description()

    def load_datasets(self, data_path, X_path, y_path):
      self.sentences = pd.read_csv(data_path + X_path)['item_caption']
      self.labels, self.one_hot_labels, self.classes = self.labels_obj.load()

    def remove_empty_description(self):
      correct_descriptions = self.sentences.notna()
      self.sentences = self.sentences[correct_descriptions].values
      self.one_hot_labels = self.one_hot_labels[correct_descriptions, :]

    def get_nb_classes(self):
      return len(self.classes)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.one_hot_labels[idx], dtype=torch.float)}