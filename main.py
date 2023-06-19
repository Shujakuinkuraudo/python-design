import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gzip
import os
import torchvision
import cv2
import matplotlib.pyplot as plt


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


trainDataset = DealDataset('dataset/MNIST/raw', "train-images-idx3-ubyte.gz",
                           "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
testDataset = DealDataset('dataset/MNIST/raw', "t10k-images-idx3-ubyte.gz",
                          "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

# 训练数据和测试数据的装载
train_loader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=10,
    shuffle=True,
)




class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()



images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
print(labels)
plt.imshow(img)
plt.show()
