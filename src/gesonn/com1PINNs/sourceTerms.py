# imports
import torch


def get_f(x, y, mu=1., name=None):
    if name == "one":
        return mu + 0*x #multiply by zero to have a tensor if mu is default value
    elif name == "ellipse":
        r2 = (x / mu) ** 2 + (mu * y) ** 2
        return torch.exp(1 - r2)
    elif name == "exp":
        r2 = (0.5 * x) ** 2 + (2 * y) ** 2
        return torch.exp(1 - r2)   
    elif name == "sin":
        r2 = (0.5 * x) ** 2 + (2 * y) ** 2
        return 5 * torch.sin(x*torch.pi)*torch.cos(y*torch.pi) * (torch.sin(x*torch.pi)*torch.cos(y*torch.pi)>0.2)
    return 0

