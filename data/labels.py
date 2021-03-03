import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class Labels:
    def __init__(self,y_path):
        self.mlb = MultiLabelBinarizer()
        self.y_path = y_path

    def load(self):
        self.labels = pd.read_csv(self.y_path, indox_col=0)
        self.labels = self.labels['color_tags'.apply(self.str_to_list)]
        self.onehot_labels = self.mlb.fit_transform(self.labels)
        self.classes_correp = self.mlb.classes_
        return self.labels, self.onehot_labels, self.classes_correp

    def str_to_list(self, s):
      s = s.replace('[', '').replace(']', '').replace(',', '')
      return s.split()
