import torch

__version__ = '0.1.0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
torch.set_default_dtype(torch.double)

print('Using device:', device)

if torch.cuda.is_available():
    print(f"cuda devices:        {torch.cuda.device_count()}")
    print(f"cuda current device: {torch.cuda.current_device()}")
    print(f"cuda device name:    {torch.cuda.get_device_name(0)}")
