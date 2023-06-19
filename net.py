import torch
class MLP(torch.nn.Module):
    def __init__(self,linear:list):
        super().__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(linear[i],linear[i+1]) for i in range(len(linear)-1)])
        self.activation = torch.nn.ReLU()
    def forward(self,x):
        for i,net in enumerate(self.linears):
            x = net(x)
            if i != len(self.linears)-1:
                x = self.activation(x)
        return x