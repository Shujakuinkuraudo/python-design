import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torchvision
import gzip
import cv2
import matplotlib.pyplot as plt
from dataset import DealDataset
from config import CFG
from net import MLP

trainDataset = DealDataset('dataset/MNIST/raw', "train-images-idx3-ubyte.gz",
                           "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
testDataset = DealDataset('dataset/MNIST/raw', "t10k-images-idx3-ubyte.gz",
                          "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

# 训练数据和测试数据的装载
train_loader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=CFG.batch,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=testDataset,
    batch_size=CFG.batch,
    shuffle=False,
)



images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1, 2, 0)
a = MLP(CFG.linear).to(CFG.device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(a.parameters(), lr=1e-3, momentum=0.9)
print(labels)
plt.imshow(img)
plt.show()


from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
def train(dataloader, model, loss_fn, optimizer):

    # Total size of dataset for reference
    size = 0

    # places your model into training mode
    model.train()

    # loss batch
    batch_loss = {}
    batch_accuracy = {}

    correct = 0
    _correct = 0



    # Gives X , y for each batch
    for batch, (X, y) in enumerate(dataloader):

        # Converting device to cuda
        X, y = X.to(CFG.device), y.to(CFG.device)
        # Compute prediction error / loss
        # 1. Compute y_pred
        # 2. Compute loss between y and y_pred using selectd loss function

        y_pred = model(X.view(-1,28*28))
        loss = loss_fn(y_pred, y)

        # Backpropagation on optimizing for loss
        # 1. Sets gradients as 0
        # 2. Compute the gradients using back_prop
        # 3. update the parameters using the gradients from step 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
        _batch_size = len(X)

        correct += _correct

        # Updating loss_batch and batch_accuracy
        batch_loss[batch] = loss.item()
        batch_accuracy[batch] = _correct/_batch_size

        size += _batch_size

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")

    correct/=size
    print(f"Train Accuracy: {(100*correct):>0.1f}%")

    return batch_loss , batch_accuracy

def validation(dataloader, model, loss_fn):

    # Total size of dataset for reference
    size = 0
    num_batches = len(dataloader)

    # Setting the model under evaluation mode.
    model.eval()

    test_loss, correct = 0, 0

    _correct = 0
    _batch_size = 0

    batch_loss = {}
    batch_accuracy = {}

    with torch.no_grad():

        # Gives X , y for each batch
        for batch , (X, y) in enumerate(dataloader):

            X, y = X.to(CFG.device), y.to(CFG.device)
            pred = model(X.view(-1,28*28))

            batch_loss[batch] = loss_fn(pred, y).item()
            test_loss += batch_loss[batch]
            _batch_size = len(X)

            _correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += _correct

            size+=_batch_size
            batch_accuracy[batch] = _correct/_batch_size




    ## Calculating loss based on loss function defined
    test_loss /= num_batches

    ## Calculating Accuracy based on how many y match with y_pred
    correct /= size

    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return batch_loss , batch_accuracy


train_batch_loss = []
train_batch_accuracy = []
valid_batch_accuracy = []
valid_batch_loss = []
train_epoch_no = []
valid_epoch_no = []

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    _train_batch_loss , _train_batch_accuracy = train(train_loader, a, loss_fn, optimizer)
    _valid_batch_loss , _valid_batch_accuracy = validation(test_loader, a, loss_fn)
    for i in range(len(_train_batch_loss)):
        train_batch_loss.append(_train_batch_loss[i])
        train_batch_accuracy.append(_train_batch_accuracy[i])
        train_epoch_no.append( t + float((i+1)/len(_train_batch_loss)))
    for i in range(len(_valid_batch_loss)):
        valid_batch_loss.append(_valid_batch_loss[i])
        valid_batch_accuracy.append(_valid_batch_accuracy[i])
        valid_epoch_no.append( t + float((i+1)/len(_valid_batch_loss)))
print("Done!")