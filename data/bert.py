from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np 
import torch

class Bert_dataset(Dataset):
    def __init__(self, X_path, y_path, max_len):
      self.sentences = pd.read_csv(X_path)['item_caption']
      correct_values = self.sentences.notna()
      self.sentences = self.sentences[correct_values].values
      self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
      self.max_len = max_len
      self.colors_dict = self.init_color_idx(y_path)
      self.targets = self.create_target_labels(y_path, self.colors_dict)
      self.targets = self.targets[correct_values, :]

    def init_color_idx(self, y_path):
      colors = {}
      df = pd.read_csv(y_path)
      for _, row in df.iterrows():
        color_list = self.str_to_list(row['color_tags'])
        for color in color_list:
          colors.setdefault(color, len(colors))
      return colors

    def create_target_labels(self, y_path, colors_dict):
      df = pd.read_csv(y_path)
      targets = np.zeros((len(df), len(colors_dict)))
      for idx, (_, row) in enumerate(df.iterrows()):
        color_list = self.str_to_list(row['color_tags'])
        for color in color_list:
          targets[idx, colors_dict[color]] = 1
      return targets

    def str_to_list(self, s):
      s = s.replace('[', '').replace(']', '').replace(',', '')
      return s.split()

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
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)}