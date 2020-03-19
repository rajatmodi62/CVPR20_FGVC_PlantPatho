import torch


def is_gpu_available():
    print("[ GPU: ", torch.cuda.get_device_name(0), " ]")
