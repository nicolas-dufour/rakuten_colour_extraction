from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from data.labels import Labels
import pandas as pd
import numpy as np 
import torch
from torch.utils.data import random_split
import random
from torch.nn.utils.rnn import pad_sequence

class Features:
  def __init__(self, ids, mask, token_type_ids, target, text_id):
    self.ids = ids
    self.mask = mask
    self.token_type_ids = token_type_ids
    self.target = target
    self.text_id = text_id
  
class Bert_dataset(Dataset):
  def __init__(self, data_path, X_path, y_path):
    self.start_token = 2
    self.end_token = 3
    self.labels_obj = Labels(data_path + y_path)
    self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    self.pad_token = self.tokenizer.pad_token
    self.pad_token_id = self.tokenizer.pad_token_id
    self.data_path = data_path
    self.X_path = X_path
    self.y_path = y_path
    self.text_size = 200
    self.overlap_size = 50
    self.build()

  def load_datasets(self, data_path, X_path, y_path):
    nb_points = 2000
    X = pd.read_csv(data_path + X_path)#.iloc[:nb_points]
    _, one_hot_labels, classes = self.labels_obj.load()
    #one_hot_labels = one_hot_labels[:nb_points, :]
    return X, one_hot_labels, classes

  
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
    inputs = self.tokenizer.encode_plus(row[0],
                                        None,
                                        add_special_tokens=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True,
                                        return_overflowing_tokens=False,
                                        return_special_tokens_mask=False)
    return Features(ids=inputs['input_ids'],
                    mask= inputs['attention_mask'],
                    token_type_ids= inputs['token_type_ids'],
                    target=row[1],
                    text_id=row[2])
                    

  def chunkenize(self, feature):
    chunks = []
    ids = feature[0].ids[1:-1]
    mask = feature[0].mask[1:-1]
    token_type = feature[0].token_type_ids[1:-1]
    starting_point = 0
    while len(ids) > starting_point:
      if starting_point == 0: # first chunk
        t = 's'
        ids_partial = ids[starting_point : starting_point + self.text_size - 2]
        mask_partial = mask[starting_point : starting_point + self.text_size - 2]
        token_type_partial = token_type[starting_point : starting_point + self.text_size - 2]
        starting_point = self.text_size - 2
      elif starting_point + self.text_size - 2 > len(ids): # last chunk
        t = 'l'
        ids_partial = ids[-(self.text_size - 2):]
        mask_partial = mask[-(self.text_size - 2):]
        token_type_partial = token_type[-(self.text_size - 2):]
        starting_point += self.text_size - 2
      else: # middle chunk
        t = 'm'
        ids_partial = ids[starting_point - self.overlap_size : starting_point + self.text_size - self.overlap_size - 2]
        mask_partial = mask[starting_point - self.overlap_size : starting_point + self.text_size - self.overlap_size - 2]
        token_type_partial = token_type[starting_point - self.overlap_size : starting_point + self.text_size - self.overlap_size - 2]
        starting_point = starting_point + self.text_size - self.overlap_size - 2
      if len(ids_partial) < self.text_size:
        ids_partial += [self.pad_token_id]*(self.text_size - len(ids_partial))
        mask_partial += [self.pad_token_id]*(self.text_size - len(mask_partial))
        token_type_partial += [1]*(self.text_size - len(token_type_partial))
      chunks.append({'ids': torch.tensor([self.start_token] + ids_partial + [self.end_token]),
                          'mask': torch.tensor([1] + mask_partial + [1]),
                          'token_type_ids': torch.tensor([0] + token_type_partial + [0]),
                          'targets': torch.tensor(feature[0].target),
                          'text_id': feature[0].text_id})
    return chunks
        



  def build(self):
    # Step 1 load
    text_df, one_hot_labels, self.classes = self.load_datasets(self.data_path, self.X_path, self.y_path)
    # Step 2 preprocess
    texts = text_df.apply(self.preprocess_text, axis=1)
    # Step 3 remove empty
    correct_text = texts.notna()
    texts = texts[correct_text]
    one_hot_labels = one_hot_labels[correct_text]
    # Step 4 Tokenize
    df = pd.DataFrame({'text': texts, 'labels': list(one_hot_labels), 'text_id': correct_text.index})
    features_list = np.apply_along_axis(self.encode, 1, df)
    # Step 5 Chunkenize
    features_list = np.expand_dims(features_list, 1)
    self.chunks = np.apply_along_axis(self.chunkenize, 1, features_list)

  def get_nb_classes(self):
    return len(self.classes)

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

  def split_dataset(self, dataset):
    size = dataset.__len__()
    train_size = int(size*0.8)
    val_size = int(size*0.1)
    test_size = size - train_size - val_size
    return random_split(dataset,[train_size, val_size, test_size])

  def plot_sizes(self, train, val, test):
    print('[SYSTEM] Train size', train.__len__())
    print('SYSTEM] Validation size', val.__len__())
    print('[SYSTEM]Test size', test.__len__())

  def build(self):
      dataset = Bert_dataset(self.data_path, self.X_path, self.y_path)
      train_dataset, val_dataset, test_dataset = self.split_dataset(dataset)
      self.plot_sizes(train_dataset, val_dataset, test_dataset)
      train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.workers)

      val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.workers)

      test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.workers)
      return train_loader, val_loader, test_loader, dataset.get_nb_classes()