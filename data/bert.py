from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from data.labels import Labels
import pandas as pd
import numpy as np 
import torch
from torch.utils.data import random_split
import random
from torch.nn.utils.rnn import pad_sequence

class Data:
  def __init__(self, t, o):
    self.text = t
    self.label = o

class Features:
  def __init__(self, ids, mask, token_type_ids, target):
    self.ids = ids
    self.mask = mask
    self.token_type_ids = token_type_ids
    self.target = target
  
class Bert_dataset(Dataset):
  def __init__(self, data_path, X_path, y_path, max_len, batch_size):
    self.batch_size = batch_size
    self.labels_obj = Labels(data_path + y_path)
    self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    self.max_len = max_len
    self.batches = self.manage_data(data_path, X_path, y_path)

  def manage_data(self, data_path, X_path, y_path):
    text_df, one_hot_labels, self.classes = self.load_datasets(data_path, X_path, y_path)
    data = self.remove_empty_description(text_df, one_hot_labels)
    data = self.sort_data(data)
    batches = self.build_batches(data)
    return batches

  def sort_data(self, data):
    data.sort(key=lambda x: x[0])
    return [x[1] for x in data]

  def load_datasets(self, data_path, X_path, y_path):
    X = pd.read_csv(data_path + X_path)
    text_df = self.preprocess_text(X['item_caption'], X['item_name'], X)
    _, one_hot_labels, classes = self.labels_obj.load()
    return text_df, one_hot_labels, classes

  def remove_empty_description(self, text_df, one_hot_labels):
    correct_texts = text_df['text'].notna()
    text_df = text_df[correct_texts]
    text_df['sizes'] = text_df['text'].apply(len)
    sizes = text_df.sizes.values
    texts = text_df['text'].values
    one_hot_labels = one_hot_labels[correct_texts, :]
    return [(s, Data(t, l)) for t, s, l in zip(texts, sizes, one_hot_labels)]

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
    return X

  def build_batches(self, data):
    batch_ordered_sentences = []
    while len(data) > 0:
        to_take = min(self.batch_size, len(data))
        select = random.randint(0, len(data) - to_take)
        batch_ordered_sentences += data[select:select + to_take]
        del data[select:select + to_take]
    return batch_ordered_sentences

  def encode(self, batch):
    inputs = self.tokenizer.encode_plus(batch.text,
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_len,
                                        truncation=True,
                                        padding='do_not_pad',
                                        return_token_type_ids=True,
                                        return_attention_mask=True,
                                        return_overflowing_tokens=False,
                                        return_special_tokens_mask=False)
    return Features(ids=inputs['input_ids'],
                    mask = inputs['attention_mask'],
                    token_type_ids = inputs["token_type_ids"],
                    target = batch.label)

  def get_nb_classes(self):
    return len(self.classes)

  def __len__(self):
    return len(self.batches)

  def __getitem__(self, idx):
    batch = self.batches[idx]
    return self.encode(batch)

def collate_batch(batch):
    ids = pad_sequence([torch.tensor(data.ids) for data in batch], True, 0).type(torch.long)
    mask = pad_sequence([torch.tensor(data.mask) for data in batch], True, 0).type(torch.long)
    token_type = pad_sequence([torch.tensor(data.token_type_ids) for data in batch], True, 0).type(torch.long)
    targets = torch.tensor([b.target for b in batch], dtype=torch.long)
    print(ids.size())
    return {"ids": ids,
            "mask": mask,
            "token_type_ids": token_type,
            "targets": targets
            }


class Bert_Data:
  def __init__(self, data_path, X_path, y_path, batch_size, workers, MAX_LEN):
    self.data_path = data_path
    self.X_path = X_path
    self.y_path = y_path
    self.batch_size = batch_size
    self.workers = workers
    self.MAX_LEN = MAX_LEN

  def split_dataset(self, dataset):
    len = dataset.__len__()
    train_size = int(len*0.8)
    val_size = int((len - train_size) / 2)
    test_size = len - train_size - val_size
    return random_split(dataset,[train_size, val_size, test_size])

  def plot_sizes(self, train, val, test):
    print('[SYSTEM] Train size', train.__len__())
    print('SYSTEM] Validation size', val.__len__())
    print('[SYSTEM]Test size', test.__len__())

  def build(self):
      dataset = Bert_dataset(self.data_path, self.X_path, self.y_path, self.MAX_LEN, self.batch_size)
      train_dataset, val_dataset, test_dataset = self.split_dataset(dataset)
      self.plot_sizes(train_dataset, val_dataset, test_dataset)

      train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.workers, 
                                collate_fn=collate_batch)

      val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.workers,
                              collate_fn=collate_batch)

      test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.workers, 
                               collate_fn=collate_batch)
      return train_loader, val_loader, test_loader, dataset.get_nb_classes()