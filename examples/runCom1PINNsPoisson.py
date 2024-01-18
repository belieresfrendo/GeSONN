# imports
import os
import torch
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

# local imports
from gesonn.com1PINNs import poisson
from gesonn.com2SympNets import G

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":
    train = True
    # train = False

    SympNetsDict = {
        "learning_rate": 1e-3,
        "nb_of_networks": 6,
        "networks_size": 10,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "test",
        "symplecto_name": None,
        "to_be_trained": False,
    }
    SympNet = G.Symp_Net(SympNetsDict=SympNetsDict)


    PINNsDict = {
        "learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 10, 1],
        "rho_min": 0.5,
        "rho_max": 1,
        "file_name": "bernoulli_ellipse",
        "symplecto_name": None,
        "SympNet": None,
        "to_be_trained": True,
    }

    epochs = 1_000
    n_collocation = 10_000
    new_training = False
    new_training = True

    if train:
        if new_training:
            try:
                os.remove("./../outputs/PINNs/net/poisson_" + PINNsDict["file_name"] + ".pth")
            except FileNotFoundError:
                pass

        network = poisson.PINNs(PINNsDict=PINNsDict)

        if device.type == "cpu":
            network.train(
                epochs=epochs, n_collocation=n_collocation, n_data=0, plot_history=True
            )
        else:
            network.train(
                epochs=epochs, n_collocation=n_collocation, n_data=0, plot_history=True
            )

    else:
        network = poisson.PINNs()

        for _ in range(1):
            network.plot_result(random=True, derivative=True)
