from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import ast
import h5py
from tqdm.notebook import tqdm
#from data.bert import Bert_dataset, Bert_dataset_Test
from torch.utils.data import Dataset, DataLoader
import torch

class Loader:
  def __init__(self, df_path_X, df_path_y, seed, label_column, one_hot_column):
    self.df_path_X = df_path_X
    self.df_path_y = df_path_y
    self.label_column = label_column
    self.seed = seed
    self.mlb = MultiLabelBinarizer()
    self.one_hot_column = one_hot_column

  def build(self):
    # Step 1 load
    df_X = pd.read_csv(self.df_path_X, index_col=0) # [index, image_file_name,item_name,item_caption]
    df_y = pd.read_csv(self.df_path_y, index_col=0) #[ index, color_tags]
    df = pd.concat([df_X, df_y], axis=1)
    # Step 2 one hot labels
    df[self.label_column] = df[self.label_column].apply(ast.literal_eval)
    onehot_labels = self.mlb.fit_transform(df[self.label_column])
    df[self.one_hot_column] = list(onehot_labels) 
    nb_classes = len(self.mlb.classes_)
    # Step 3 split
    np.random.seed(self.seed)
    idx = np.random.permutation(len(df))
    sep = int(len(df)*0.9)
    idx_train, idx_val = idx[:sep], idx[sep:]
    df_train, df_val = df.iloc[idx_train], df.iloc[idx_val]
    return df_train, df_val, nb_classes

# def save_chunks_bert(dataloader, filename, model, device):
#   model.eval()
#   table_size = (dataloader.dataset.__len__(), 768)
#   idx_init = 0
#   with torch.no_grad():
#     with h5py.File(filename, "w") as f:
#         embedding_dataset = f.create_dataset("embedding", (table_size), dtype='f')
#         for idx, (ids, mask, _, text_id, chunk_id) in enumerate(tqdm(dataloader)):
#           ids = ids.to(device)
#           mask = mask.to(device)
#           embeddings = model.input_layer(ids, attention_mask = mask, token_type_ids = None).pooler_output.detach().cpu().numpy()
#           idx_end = idx_init + embeddings.shape[0]
#           embedding_dataset[idx_init:idx_end] = embeddings

#         f.close()     

# def save_chunk_text_id(dataset, batch_size, filename):
#   ci, ti = 0, 0
#   df = df = pd.DataFrame({'text_id': [], 'chunk_init': [], 'chunk_end': []})
#   for i, (_, _, _, text_id, chunk_id) in enumerate(tqdm(dataset)):
#     if chunk_id == 0 and i != 0:
#       ce = int(i - 1)
#       df = df.append({'text_id': ti, 'chunk_init': ci, 'chunk_end': ce}, ignore_index=True)
#       ci = int(i)
#       ti = int(text_id)
#   ce = int(i)
#   df = df.append({'text_id': ti, 'chunk_init': ci, 'chunk_end': ce}, ignore_index=True)
#   df.to_csv(filename)

# def save_bert_train_val(df_path_X, df_path_y, filename_train, filename_val, model, csv_train_file, csv_val_file, batch_size): 
#   train_df, val_df, _ = Loader(df_path_X, df_path_y, 42, 'color_tags', 'one_hot').build()
#   train_set = Bert_dataset(train_df)
#   train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=8)
#   val_set = Bert_dataset(val_df)
#   val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
#   save_chunk_text_id(train_set, batch_size, csv_train_file)
#   save_chunks_bert(train_loader, filename_train, model, model.device)
#   save_chunk_text_id(val_set, batch_size, csv_val_file)
#   save_chunks_bert(val_loader, filename_val, model, model.device)

# def save_bert_test(df_path_X, batch_size, model, filename_test, csv_test_file):
#   test_set = Bert_dataset_Test(df_path_X)
#   test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
#   save_chunk_text_id(test_set, batch_size, csv_test_file)
#   save_chunks_bert(test_loader, filename_test, model, model.device)

