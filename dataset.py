from torch.utils.data import  Dataset
import gzip
import numpy as np
import os
class DealDataset(Dataset):

    def __init__(self, folder, data_name, label_name, transform=None):
        self.train_set, self.train_labels = self.load_data(folder, data_name, label_name)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

    @staticmethod
    def load_data(data_folder, data_file, label_file):
        with gzip.open(os.path.join(data_folder, label_file), 'rb') as labelpath, gzip.open(
                os.path.join(data_folder, data_file), 'rb') as datapath:
            label = np.frombuffer(labelpath.read(), dtype=np.uint8, offset=8)
            data = np.frombuffer(datapath.read(), dtype=np.uint8, offset=16).reshape(len(label), 28, 28)
        return np.array(data), np.array(label)