# imports
import torch


def apply_symplecto(x, y, mu=0.5, name=None, SympNet=None):
    if SympNet == None:
        if name == "bizaroid":
            x = x - 0.5 * y * y + 0.3 * torch.sin(2.0 * y) - 0.2 * torch.sin(8.0 * y)
            y = y + 0.1 * x + 0.12 * torch.cos(x)
        if name == "avocado":
            x = x - 0.5 * y**2 + 0.3 * torch.sin(y)
            y = y + 0.1 * x + 0.12 * torch.cos(x)
        if name == "galaxy":
            x = x - 0.5 * y
            y = y + 0.1 * x
        if name == "ellipse":
            x = 1/mu * x + y - y
            y = mu * y + x - x
    elif SympNet != None:
        x, y = SympNet.apply_symplecto(x, y, mu)
    return x, y
