import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange

import wandb
from config import CFG_sweep
from net import MLP
from utils import MNIST, wandb_init_sweep

sweep_configuration = {
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'val_correct'},
    'parameters':
        {
            'batch_size': {'values': [1, 128, 64]
                           },
            'linear': {'values': [[784, 2500, 2000, 1500, 1000, 500, 10],
                                  [28 * 28, 10000, 10],
                                  [28 * 28, 14 * 14, 14 * 14, 14 * 14, 10],
                                  [28 * 28, 14 * 14, 14 * 14, 10]]
                       },
            'optim_optim_config': {'values': [("torch.optim.Adam", {"lr": 1e-3}),
                                              ("torch.optim.SGD", {"lr": 1e-3, "momentum": 0.9}),
                                              ("torch.optim.Adadelta", {"lr": 1e-2}),
                                              ("torch.optim.Adagrad", {"lr": 1e-2}),
                                              ("torch.optim.Adamax", {"lr": 1e-3}),
                                              ("torch.optim.RMSprop", {"lr": 1e-2}),
                                              ]
                                   },
            # 'swa_model': {'values':["torch.optim.swa_utils.AveragedModel"]
            # },
            'scheduler_scheduler_config': {'values': [(None, None),
                                                      ("torch.optim.lr_scheduler.CosineAnnealingLR", {"T_max": 150}),
                                                      ("torch.optim.lr_scheduler.StepLR", {"step_size": 30}),
                                                      ("torch.optim.lr_scheduler.ExponentialLR", {"gamma": 0.999}),
                                                      ]
                                           },

            'bn': {'values': [True, False]},
            'dp': {'values': [0, 0.1, 0.2, 0.3, 0.4]}
        }
}


def main():
    run = wandb_init_sweep(CFG_sweep)
    trainDataset = MNIST('dataset/MNIST/raw', "train-images-idx3-ubyte.gz",
                         "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
    testDataset = MNIST('dataset/MNIST/raw', "t10k-images-idx3-ubyte.gz",
                        "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

    # 训练数据和测试数据的装载
    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=run.config.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=run.config.batch_size,
        shuffle=False,
    )

    model = MLP(run.config.linear, run.config.bn, run.config.dp).to(run.config.device)

    loss_fn = eval(run.config.lossfn)()

    optim, optim_config = run.config.optim_optim_config
    optimizer = eval(optim)(model.parameters(), **optim_config)

    # swa = eval(run.config.swa_model)(model)

    scheduler, scheduler_config = run.config.scheduler_scheduler_config
    if scheduler:
        scheduler = eval(scheduler)(optimizer, **scheduler_config)

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
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            X, y = X.to(run.config.device), y.to(run.config.device)
            y_pred = model(X.view(-1, 28 * 28))
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            batch_correct.append((y_pred.argmax(1) == y).sum().item() / len(X))
            size += len(X)
        train_loss = sum(batch_loss) / len(batch_loss)
        train_correct = sum(batch_correct) / len(batch_correct)
        # tqdm.write(f"Train Error: \t Accuracy: {(100*train_correct):>0.5f}%, Avg loss: {train_loss:>8f} ")
        return train_correct, train_loss

    def validation(dataloader, model, loss_fn):
        size = 0
        model.eval()
        batch_loss = []
        batch_correct = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(run.config.device), y.to(run.config.device)
                pred = model(X.view(-1, 28 * 28))
                batch_loss.append(loss_fn(pred, y).item())
                batch_correct.append((pred.argmax(1) == y).sum().item() / len(X))
                size += len(X)
        test_loss = sum(batch_loss) / len(batch_loss)
        test_correct = sum(batch_correct) / len(batch_correct)
        return test_correct, test_loss

    final_val_correct = 0
    correct_max, correct_max_iter = 0, 0
    with trange(run.config.epochs) as t:
        for _ in t:
            train_accuracy, train_loss = train(train_loader, model, loss_fn, optimizer)
            correct, test_loss = validation(test_loader, model, loss_fn)
            if correct > correct_max:
                correct_max = correct
                correct_max_iter = _
            if scheduler:
                scheduler.step()
            if run.config.wandb:
                run.log({"epoch": _, "train_correct": train_accuracy * 100, "train_loss": train_loss,
                         "val_correct": correct * 100, "test_loss": test_loss})
            t.set_postfix(train_correct=train_accuracy * 100, train_loss=train_loss, val_correct=correct * 100,
                          test_loss=test_loss)
            final_val_correct = correct
    if run.config.wandb:
        run.log({"correct_max": correct_max, "correct_max_iter": correct_max_iter})
        run.finish()
        torch.save(model.state_dict(), f"weights/{run.config.linear}-{final_val_correct}.pt")


sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='python-hw'
)

wandb.agent(sweep_id, function=main)