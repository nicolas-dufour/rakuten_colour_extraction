from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from data.labels import Labels
import pandas as pd
import numpy as np 
import torch
from torch.utils.data import random_split

class Bert_dataset(Dataset):
  def __init__(self, data_path, X_path, y_path, max_len):
    self.labels_obj = Labels(data_path + y_path)
    self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    self.max_len = max_len
    self.load_datasets(data_path, X_path, y_path)
    self.remove_empty_description()

  def load_datasets(self, data_path, X_path, y_path):
    X = pd.read_csv(data_path + X_path)
    self.texts = self.preprocess_text(X['item_caption'], X['item_name'], X)
    self.labels, self.one_hot_labels, self.classes = self.labels_obj.load()

  def remove_empty_description(self):
    correct_texts = self.texts.notna()
    self.texts = self.texts[correct_texts].values
    self.one_hot_labels = self.one_hot_labels[correct_texts, :]

  def preprocess_text(self, sentences, titles, X):
    correct_descriptions = sentences.notna()
    correct_titles = titles.notna()
    unique_description = np.logical_and(correct_descriptions, correct_titles == False)
    unique_title = np.logical_and(correct_descriptions == False, correct_titles)
    both = np.logical_and(correct_descriptions, correct_titles)
    text_both = "<title> " + titles[both] + '. <description> ' + sentences[both]
    text_title = "<title> " + titles[unique_title] + '. <description> ' 
    text_description = "<title> . <description> " +sentences[unique_description]
    X['text'] = np.nan
    X['text'].loc[unique_description] = text_description
    X['text'].loc[unique_title] = text_title
    X['text'].loc[both] = text_both
    return X['text']

  def get_nb_classes(self):
    return len(self.classes)

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    sentence = self.texts[idx]
    inputs = self.tokenizer.encode_plus(
      sentence,
      None,
      add_special_tokens=True,
      max_length=self.max_len,
      padding='max_length',
      return_token_type_ids=True,
      truncation=True)
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]
    return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'targets': torch.tensor(self.one_hot_labels[idx], dtype=torch.float)}



# def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
#     # IRL, use pad_sequence
#     # https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pad_sequence.html
#     return seq + (max_batch_len - len(seq)) * [pad_value]@dataclass

# class SmartCollator(DataCollator):
#     pad_token_id: int    
#     def collate_batch(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
#         batch_inputs = list()
#         batch_attention_masks = list()
#         labels = list()
#         # find the max length of the mini batch
#         max_size = max([len(ex.input_ids) for ex in batch])
#         for item in batch:
#             # apply padding at the mini batch level
#             batch_inputs += [pad_seq(item.input_ids, max_size, self.pad_token_id)]
#             batch_attention_masks += [pad_seq(item.attention_mask, max_size, 0)]
#             labels.append(item.label)
#         # expected Transformers input format (dict of Tensors)
#         return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
#                 "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
#                 "labels": torch.tensor(labels, dtype=torch.long)
#                 }


class Bert_Data:
  def __init__(self, data_path, X_path, y_path, batch_size, workers):
    self.data_path = data_path
    self.X_path = X_path
    self.y_path = y_path
    self.batch_size = batch_size
    self.workers = workers

    def split_dataset(self, dataset):
      len = dataset.__len__()
      train_size = int(len*0.8)
      val_size = int((len - train_size) / 2)
      test_size = len - train_size - val_size
      train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size])

  def plot_sizes(self, train, val, test):
    print('[SYSTEM] Train size', train.__len__())
    print('SYSTEM] Validation size', val.__len__())
    print('[SYSTEM]Test size', test.__len__())

  def build(self):
      dataset = Bert_dataset(data_path, X_path, y_path, MAX_LEN)
      train_dataset, val_dataset, test_dataset = self.split_dataset(dataset)
      self.plot_sizes(train_dataset, val_dataset, test_dataset)

      train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.workers)

      val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.workers)

      test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.workers)
      return train_loader, val_loader, test_loader