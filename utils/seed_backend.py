import torch
import numpy
import random

def seed_all( seed ):
    if not seed:
        seed = 10
    
    print("[ Using Seed : ", seed," ]")
    
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False