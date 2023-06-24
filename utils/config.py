from easydict import EasyDict


import torch
CFG_MLP = EasyDict()
CFG_MLP.batch_size = 64
CFG_MLP.linear = [784, 2500, 2000, 1500, 1000, 500, 10]
CFG_MLP.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG_MLP.epochs = 300
CFG_MLP.optim = "torch.optim.Adam"
CFG_MLP.optim_config = {"lr":1e-3}
CFG_MLP.lossfn = torch.nn.CrossEntropyLoss()
CFG_MLP.project = "python-CourseDesign"
CFG_MLP.wandb = False
CFG_MLP.bn = True
CFG_MLP.dp = 0.3
CFG_MLP.__optim_function = lambda parameter: eval(CFG_MLP.optim)(parameter, **CFG_MLP.optim_config)


CFG_sweep = EasyDict()
CFG_sweep.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG_sweep.epochs = 150
CFG_sweep.lossfn = "torch.nn.CrossEntropyLoss"
CFG_sweep.project = "python-CourseDesign"
CFG_sweep.wandb = True


CFG_RES = EasyDict()
CFG_RES.batch_size = 64
CFG_RES.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG_RES.epochs = 50
CFG_RES.optim = "torch.optim.Adam"
CFG_RES.optim_config = {"lr":1e-3}
CFG_RES.lossfn = torch.nn.CrossEntropyLoss()
CFG_RES.project = "python-hw"
CFG_RES.wandb = True
CFG_RES.__optim_function = lambda parameter: eval(CFG_RES.optim)(parameter, **CFG_RES.optim_config)

