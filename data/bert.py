from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from data.labels import Labels
import pandas as pd
import numpy as np 
import torch
from torch.utils.data import random_split
import random
from data.commun import Loader 

class Features:
  def __init__(self, ids, mask, target, text_id):
    self.ids = ids
    self.mask = mask
    self.target = target
    self.text_id = text_id
  
class Bert_dataset(Dataset):
  def __init__(self, df):
    self.start_token = 2
    self.end_token = 3
    self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    self.pad_token = self.tokenizer.pad_token
    self.pad_token_id = self.tokenizer.pad_token_id
    self.text_size = 200
    self.overlap_size = 50
    self.df = df
    self.build()


  
  def preprocess_text(self, row):
    description = row['item_caption']
    title = row['item_name']
    if type(description) == str:
      if type(title) == str:
        text = f"<title> {title}. <description> {description}" 
      else:
        text = f"<title> <empty>. <description> {description}"
    else:
      if type(title) == str:
        text = f"<title> {title}. <description> <empty>"
      else:
        text = np.nan
    return text

  def encode(self, row):
    inputs = self.tokenizer.encode_plus(row[-2],
                                        None,
                                        add_special_tokens=True,
                                        return_token_type_ids=False,
                                        return_attention_mask=True,
                                        return_overflowing_tokens=False,
                                        return_special_tokens_mask=False)
    return Features(ids=inputs['input_ids'],
                    mask= inputs['attention_mask'],
                    target=row[-3],
                    text_id=row[-1])
                    

  def chunkenize(self, feature):
    ids = feature[0].ids[1:-1]
    mask = feature[0].mask[1:-1]
    starting_point = 0
    while len(ids) > starting_point:
      if starting_point == 0: # first chunk
        ids_partial = ids[starting_point : starting_point + self.text_size - 2]
        mask_partial = mask[starting_point : starting_point + self.text_size - 2]
        starting_point = self.text_size - 2
      elif starting_point + self.text_size - 2 > len(ids): # last chunk
        ids_partial = ids[-(self.text_size - 2):]
        mask_partial = mask[-(self.text_size - 2):]
        starting_point += self.text_size - 2
      else: # middle chunk
        ids_partial = ids[starting_point - self.overlap_size : starting_point + self.text_size - self.overlap_size - 2]
        mask_partial = mask[starting_point - self.overlap_size : starting_point + self.text_size - self.overlap_size - 2]
        starting_point = starting_point + self.text_size - self.overlap_size - 2
      if len(ids_partial) < self.text_size:
        ids_partial += [self.pad_token_id]*(self.text_size - len(ids_partial))
        mask_partial += [self.pad_token_id]*(self.text_size - len(mask_partial))
      self.chunks.append((torch.tensor([self.start_token] + ids_partial + [self.end_token]).long(),
                          torch.tensor([1] + mask_partial + [1]).long(),
                          torch.tensor(feature[0].target).long(),
                          feature[0].text_id))

  def build(self):
    # Step 1 preprocess text
    self.df['text'] = self.df.apply(self.preprocess_text, axis=1)
    # Step 3 remove empty
    correct_text = self.df['text'].notna()
    self.df = self.df[correct_text]
    self.df['text_id'] = self.df.index
    # Step 4 Tokenize
    features_list = np.apply_along_axis(self.encode, 1, self.df)
    # Step 5 Chunkenize
    features_list = np.expand_dims(features_list, 1)
    self.chunks = []
    np.apply_along_axis(self.chunkenize, 1, features_list)
    del features_list, correct_text, self.df

  def __len__(self):
    return len(self.chunks)

  def __getitem__(self, idx):
    return self.chunks[idx]

class Bert_Data:
  def __init__(self, data_path, X_path, y_path, batch_size, workers):
    self.data_path = data_path
    self.X_path = X_path
    self.y_path = y_path
    self.batch_size = batch_size
    self.workers = workers

  def plot_sizes(self, train, val):
    print('[SYSTEM] Train size', train.__len__())
    print('SYSTEM] Validation size', val.__len__())

  def build(self):
      df_path_X = self.data_path + self.X_path
      df_path_y = self.data_path + self.y_path
      train_df, val_df, nb_classes = Loader(df_path_X, df_path_y, 42, 'color_tags', 'one_hot').build()
      # train_df = train_df.sample(100)
      # val_df = val_df.sample(100)
      train_set = Bert_dataset(train_df)
      val_set = Bert_dataset(val_df)
      self.plot_sizes(train_set, val_set)
      train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.workers)

      val_loader = DataLoader(val_set, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.workers)
      return train_loader, val_loader, nb_classes