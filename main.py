import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from utils import MNIST,wandb_init
from config import CFG_CNN
from net import MLP,CNN,RES
from tqdm import tqdm,trange
import numpy as np
import itertools

def main(CFG):
    trainDataset = MNIST('dataset/MNIST/raw', "train-images-idx3-ubyte.gz",
                               "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
    testDataset = MNIST('dataset/MNIST/raw', "t10k-images-idx3-ubyte.gz",
                              "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

    # 训练数据和测试数据的装载
    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=CFG.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=CFG.batch_size,
        shuffle=False,
    )


    # images, labels = next(iter(train_loader))
    # img = torchvision.utils.make_grid(images)
    # img = img.numpy().transpose(1, 2, 0)
    # print(labels)
    # plt.imshow(img)
    # plt.show()

    model = eval(CFG.model)()
    # 修改全连接层的输出
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model.to(CFG.device)

    loss_fn = CFG.lossfn
    optimizer = CFG.__optim_function(model.parameters())

    if CFG.wandb:
        run = wandb_init(CFG)
        run.watch(model, log='all')
        Total_params = 0
        Trainable_params = 0
        for param in model.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
        run.log({"Total params": Total_params, "Trainable params": Trainable_params})

    def train(dataloader, model, loss_fn, optimizer):
        size = 0
        model.train()
        batch_loss = []
        batch_correct = []
        for i,(X, y) in enumerate(dataloader):
            X, y = X.to(CFG.device), y.to(CFG.device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            batch_correct.append((y_pred.argmax(1) == y).sum().item() / len(X))
            size += len(X)
        train_loss = sum(batch_loss) / len(batch_loss)
        train_correct = sum(batch_correct) / len(batch_correct)
        # tqdm.write(f"Train Error: \t Accuracy: {(100*train_correct):>0.5f}%, Avg loss: {train_loss:>8f} ")
        return train_correct,train_loss

    def validation(dataloader, model, loss_fn):
        size = 0
        model.eval()
        batch_loss = []
        batch_correct = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(CFG.device), y.to(CFG.device)
                pred = model(X)
                batch_loss.append(loss_fn(pred, y).item())
                batch_correct.append((pred.argmax(1) == y).sum().item() / len(X))
                size += len(X)
        test_loss = sum(batch_loss) / len(batch_loss)
        test_correct = sum(batch_correct) / len(batch_correct)
        return test_correct,test_loss


    correct_max, correct_max_iter = 0, 0
    final_testloss = 0
    with trange(CFG.epochs) as t:
        for _ in t:
            train_accuracy,train_loss = train(train_loader, model, loss_fn, optimizer)
            correct,test_loss = validation(test_loader, model, loss_fn)
            if correct > correct_max:
                correct_max = correct
                correct_max_iter = _
            if CFG.wandb:
                run.log({"epoch": _, "train_correct": train_accuracy*100, "train_loss": train_loss, "val_correct": correct*100, "test_loss": test_loss})
            t.set_postfix(train_correct=train_accuracy*100, train_loss=train_loss, val_correct=correct*100, test_loss=test_loss)
            final_testloss = test_loss
    if CFG.wandb:
        run.log({"correct_max": correct_max, "correct_max_iter": correct_max_iter})
        run.finish()
        torch.save(model.state_dict(), f"weights/{CFG.model}-{final_testloss}.pt")


for model in ["torchvision.models.resnet18","torchvision.models.resnet50","torchvision.models.resnet101","torchvision.models.resnet152"]:
    CFG_CNN.model = model
    main(CFG_CNN)