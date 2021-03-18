from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import ast
import h5py
from tqdm.notebook import tqdm

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


def extract_and_save_bert(dataloader, embedding_size, model, device, filename):
  table_size = (dataloader.dataset.nb_texts, dataloader.dataset.nb_chunks_max, embedding_size)
  with h5py.File(filename, "w") as f:
        embedding_dataset = f.create_dataset("embedding", (table_size), dtype='f')
        for idx, (ids, mask, _, text_id, chunk_id) in enumerate(tqdm(dataloader)):
            ids = ids.to(device)
            mask = mask.to(device)
            embeddings = model.input_layer(ids, attention_mask = mask, token_type_ids = None).pooler_output.detach().cpu().numpy()
            for t_id, c_id, emb in zip(text_id.numpy(), chunk_id.numpy(), embeddings):
              embedding_dataset[t_id, c_id] = emb
            if idx % 100 == 99:
                f.flush()
        f.close()