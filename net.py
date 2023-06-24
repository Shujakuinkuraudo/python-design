import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self,linear:list,normalize=False,dropout = 0):
        super().__init__()
        self.normalize = normalize
        self.dropout=nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(linear[0])
        self.linears = nn.ModuleList([torch.nn.Linear(linear[i],linear[i+1]) for i in range(len(linear)-1)])
        self.activation = nn.ReLU()
    def forward(self,x):
        x = torch.flatten(x, 1)
        if self.normalize:
            x = self.bn(x)
        for i,net in enumerate(self.linears):
            x = self.dropout(x)
            x = net(x)
            if i != len(self.linears)-1:
                x = self.activation(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5,padding="same")
        self.conv2 = nn.Conv2d(6, 16, 5,padding="same")
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


