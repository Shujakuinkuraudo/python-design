import torch
class MLP(torch.nn.Module):
    def __init__(self,linear:list,normalize=False,dropout = 0):
        super().__init__()
        self.normalize = normalize
        self.dropout=torch.nn.Dropout(p=dropout)
        self.bn = torch.nn.BatchNorm1d(linear[0])
        self.linears = torch.nn.ModuleList([torch.nn.Linear(linear[i],linear[i+1]) for i in range(len(linear)-1)])
        self.activation = torch.nn.ReLU()
    def forward(self,x):
        if self.normalize:
            x = self.bn(x)
        for i,net in enumerate(self.linears):
            x = self.dropout(x)
            x = net(x)
            if i != len(self.linears)-1:
                x = self.activation(x)
        return x