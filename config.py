from easydict import EasyDict


import torch
CFG_CNN = EasyDict()
CFG_CNN.batch_size = 64
CFG_CNN.linear = [784, 2500, 2000, 1500, 1000, 500, 10]
CFG_CNN.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG_CNN.epochs = 300
CFG_CNN.optim = "torch.optim.Adam"
CFG_CNN.optim_config = {"lr":1e-3}
CFG_CNN.lossfn = torch.nn.CrossEntropyLoss()
CFG_CNN.project = "python-CourseDesign"
CFG_CNN.wandb = False
CFG_CNN.bn = True
CFG_CNN.dp = 0.3
CFG_CNN.__optim_function = lambda parameter: eval(CFG_CNN.optim)(parameter, **CFG_CNN.optim_config)


CFG_sweep = EasyDict()
CFG_sweep.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG_sweep.epochs = 150
CFG_sweep.lossfn = "torch.nn.CrossEntropyLoss"
CFG_sweep.project = "python-CourseDesign"
CFG_sweep.wandb = True


import torch
CFG_CNN = EasyDict()
CFG_CNN.batch_size = 64
CFG_CNN.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG_CNN.epochs = 50
CFG_CNN.optim = "torch.optim.Adam"
CFG_CNN.optim_config = {"lr":1e-3}
CFG_CNN.lossfn = torch.nn.CrossEntropyLoss()
CFG_CNN.project = "python-hw"
CFG_CNN.wandb = True
CFG_CNN.__optim_function = lambda parameter: eval(CFG_CNN.optim)(parameter, **CFG_CNN.optim_config)

