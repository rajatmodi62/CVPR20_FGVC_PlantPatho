import torch


def get_training_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("[ GPU: ", torch.cuda.get_device_name(0), " ]")
    else:
        device = torch.device("cpu")
        print("[ Running on the CPU ]")

    return device