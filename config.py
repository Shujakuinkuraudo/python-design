from easydict import EasyDict
import torch
CFG = EasyDict()
CFG.batch = 128
CFG.linear = [28*28,14*14,14*14,14*14,10]
CFG.device = "cuda" if torch.cuda.is_available() else "cpu"