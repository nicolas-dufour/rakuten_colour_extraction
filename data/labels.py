import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import ast

class Labels:
    def __init__(self,y_path):
        self.mlb = MultiLabelBinarizer()
        self.y_path = y_path

    def load(self):
        self.labels = pd.read_csv(self.y_path, index_col=0)
        self.labels = self.labels['color_tags'].apply(ast.literal_eval)
        self.onehot_labels = self.mlb.fit_transform(self.labels)
        self.classes_correp = self.mlb.classes_
        return self.labels, self.onehot_labels, self.classes_correp