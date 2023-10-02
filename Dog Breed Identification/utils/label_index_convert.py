import pandas as pd
import os


class LabelIndexConvert():
    def __init__(self, root_path):
        self.root_path = root_path
        self.label_name = pd.read_csv(os.path.join(root_path, 'data', 'label_index.csv'))
        self.label_to_index = self.get_label_to_index()
        self.index_to_label = self.get_index_to_label()

    def get_label_to_index(self):
        label_to_index = {}
        for i in range(len(self.label_name)):
            label_to_index[self.label_name.iloc[i, 0]] = i
        return label_to_index

    def get_index_to_label(self):
        index_to_label = {}
        for i in range(len(self.label_name)):
            index_to_label[i] = self.label_name.iloc[i, 0]
        return index_to_label

    def __getitem__(self, item):
        print(item)
        if isinstance(item, str):
            return self.label_to_index[item]
        else:
            print(item)
            return self.index_to_label[item]

if __name__ == '__main__':
    root_path = os.getcwd()
    # 返回上一级目录
    root_path = os.path.dirname(root_path)
    label_index_convert = LabelIndexConvert(root_path)
    print(label_index_convert['affenpinscher'])
    print(label_index_convert[0])
