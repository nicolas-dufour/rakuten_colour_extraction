from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import ast

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