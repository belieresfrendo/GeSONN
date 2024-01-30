# imports
import torch


def apply_symplecto(x, y, mu=0.5, name=None):
    if name == "bizaroid":
        x = x - mu * y * y + 0.3 * torch.sin(1/mu * y) - 0.2 * torch.sin(8.0 * y)
        y = y + 0.2*mu * x + 0.12 * torch.cos(x)
    if name == "avocado":
        x = x - mu * y**2 + 0.3 * torch.sin(y)
        y = y + 0.2*mu * x + 0.12 * torch.cos(x)
    if name == "galaxy":
        x = x - mu * y
        y = y + 0.2*mu * x
    if name == "ellipse":
        x = 1 / mu * x + y - y
        y = mu * y + x - x
    if name == "ellipse_benchmark":
        x = 0.5 * x + y - y
        y = 2 * y + x - x
    if name == "inverse_ellipse_benchmark":
        x = 2 * x + y - y
        y = 0.5 * y + x - x
    return x, y
