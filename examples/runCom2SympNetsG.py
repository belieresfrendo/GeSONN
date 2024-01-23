# imports
import os
import torch
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

# local imports
from gesonn.com2SympNets import G

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":
    train = True
    # train = False

    SympNetsDict = {
        "learning_rate": 1e-2,
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "default",
        "symplecto_name": "ellipse",
        "to_be_trained": True,
    }

    epochs = 1
    n_collocation = 100_000
    new_training = False
    # new_training = True

    if train:
        if new_training:
            try:
                os.remove(
                    "./../outputs/SympNets/net/" + SympNetsDict["file_name"] + ".pth"
                )
            except FileNotFoundError:
                pass

        network = G.Symp_Net(SympNetsDict=SympNetsDict)

        if device.type == "cpu":
            network.train(
                epochs=epochs, n_collocation=n_collocation, plot_history=True
            )
        else:
            network.train(
                epochs=epochs, n_collocation=n_collocation, plot_history=True
            )

    else:
        network = G.Symp_Net()

        for _ in range(1):
            network.plot_result(random=True, derivative=True)
