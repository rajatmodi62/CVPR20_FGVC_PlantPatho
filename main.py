import torch

def main():
	print(torch.cuda.get_device_name(0))

if __name__ == '__main__':
    main()
