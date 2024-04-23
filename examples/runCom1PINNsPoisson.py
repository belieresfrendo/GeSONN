# imports
import os

import torch

# local imports
from gesonn.com1PINNs import poisson

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":

    # ==============================================================
    # Parameters to be modified freely by the user
    # ==============================================================

    train = True
    # train = False

    PINNsDict = {
        "learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 80, 20, 10, 1],
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "default",
        "symplecto_name": "bizaroid",
        "to_be_trained": train,
        "source_term": "ellipse",
        "boundary_condition": "homogeneous_dirichlet",
    }

    epochs = 2000
    n_collocation = 10_000
    new_training = False
    new_training = True
    save_plots = False
    save_plots = True
    
    # ==============================================================
    # End of the modifiable area
    # ==============================================================

    if train:
        if new_training:
            try:
                os.remove(
                    "./../outputs/PINNs/net/" + PINNsDict["file_name"] + ".pth"
                )
            except FileNotFoundError:
                pass

        network = poisson.PINNs(PINNsDict=PINNsDict)

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
        network = poisson.PINNs()
        network.plot_result()
