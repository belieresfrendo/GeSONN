# imports
import os

import torch

# local imports
from gesonn.com1PINNs import bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":

    # ==============================================================
    # Parameters to be modified freely by the user
    # ==============================================================

    train = True
    # train = False

    PINNsDict = {
        "learning_rate": 1e-3,
        "layer_sizes": [2, 10, 40, 40, 10, 1],
        "rho_min": 0.5,
        "rho_max": 1,
        "a": 0.5,
        "file_name": "bernoulli_default",
        "symplecto_name": None,
        "to_be_trained": train,
        "source_term": "zero",
        "boundary_condition": "bernoulli",
    }

    epochs = 2_000
    n_collocation = 5_000
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

        network = bernoulli.PINNs(PINNsDict=PINNsDict)

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
        network = bernoulli.PINNs()
        network.plot_result()
