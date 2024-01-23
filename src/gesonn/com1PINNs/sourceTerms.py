# imports
import torch


def get_f(x, y, mu=1, name=None):
    if name == "one":
        return 1. + 0*x
    elif name == "ellipse":
        r2 = (x / 0.8) ** 2 + (0.8 * y) ** 2
        return torch.exp(1 - r2)
    elif name == "exp":
        r2 = (0.5 * x) ** 2 + (2 * y) ** 2
        return torch.exp(1 - r2)   
    return 0

