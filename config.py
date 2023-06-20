from easydict import EasyDict


import torch
CFG = EasyDict()
CFG.batch_size = 128
CFG.linear = [28*28,14*14,14*14,14*14,10]
CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG.epochs = 150
CFG.optim = "torch.optim.SGD"
CFG.optim_config = {"lr":1e-3, "momentum":0.9}
CFG.lossfn = torch.nn.CrossEntropyLoss()
CFG.project = "python-CourseDesign"
CFG.wandb = True
CFG.bn = True
CFG.__optim_function = lambda parameter: eval(CFG.optim)(parameter,**CFG.optim_config)


CFG_sweep = EasyDict()
CFG_sweep.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG_sweep.epochs = 150
CFG_sweep.lossfn = "torch.nn.CrossEntropyLoss"
CFG_sweep.project = "python-CourseDesign"
CFG_sweep.wandb = True