from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms


class Dog_dataset(Dataset):
    def __init__(self, root_path, mode='train', transform=None, label_index_convert=None):

        self.mode = mode
        self.root_path = root_path
        self.labels = pd.read_csv(os.path.join(root_path, "data", 'labels.csv'))
        self.unread_data = self.get_img_info()
        self.label_index_convert = label_index_convert

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])
            ])

    def __getitem__(self, idx):
        img_name = self.unread_data[idx]
        img_path = os.path.join(self.root_path, "data", self.mode, img_name)
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.mode == 'train':
            # 获取标签
            try:
                label = self.labels[self.labels['id'] == img_name.split('.')[0]]['breed'].item()
            except ValueError:
                print(img_name)
                label = self.labels[self.labels['id'] == img_name.split('.')[0]]['breed'].values[0]
            label = torch.as_tensor(self.label_index_convert[label], dtype=torch.long)
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.unread_data)

    def get_img_info(self):
        root_path = self.root_path
        data_path = os.path.join(root_path, "data", self.mode)
        img_info = os.listdir(data_path)
        return img_info


if __name__ == '__main__':
    root_path = os.getcwd()
    train_dataset = Dog_dataset(root_path)
