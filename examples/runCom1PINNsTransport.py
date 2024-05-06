# imports
import os

import torch

# local imports
from gesonn.com1PINNs import transport

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":

    # ==============================================================
    # Parameters to be modified freely by the user
    # ==============================================================

    train = True
    train = False

    PINNsDict = {
        "learning_rate": 1e-3,
        "layer_sizes": [2, 10, 20, 40, 40, 20, 10, 1],
        "file_name": "transport",
        "to_be_trained": train,
        "boundary_condition": "homogeneous_dirichlet",
    }

    epochs = 50_000
    n_collocation = 1_000
    new_training = False
    new_training = True
    save_plots = False
    # save_plots = True

    # ==============================================================
    # End of the modifiable area
    # ==============================================================

    if train:
        if new_training:
            try:
                print("./../outputs/PINNs/net/" + PINNsDict["file_name"] + ".pth")
                os.remove(
                    "./../outputs/PINNs/net/" + PINNsDict["file_name"] + ".pth"
                )
            except FileNotFoundError:
                pass

        network = transport.PINNs(PINNsDict=PINNsDict)

        if device.type == "cpu":
            tps = network.train(
                epochs=epochs,
                n_collocation=n_collocation,
                plot_history=True,
                save_plots=save_plots,
            )
        else:
            tps = network.train(
                epochs=epochs,
                n_collocation=n_collocation,
                plot_history=True,
                save_plots=save_plots,
            )
        print(f"Computational time: {str(tps)[:4]} sec.")

    else:
        network = transport.PINNs(PINNsDict=PINNsDict)
        network.plot_result()
