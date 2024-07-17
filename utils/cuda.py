import torch


def get_device(device_name="cuda:0"):
    print("Cuda available:", torch.cuda.is_available())
    print ('Available devices ', torch.cuda.device_count())
    dev = torch.device(device_name) if torch.cuda.is_available() else torch.device("cpu")
    print ('Current cuda device: {} ({})'.format(torch.cuda.current_device(), torch.cuda.get_device_name(dev)))
    return(dev)
