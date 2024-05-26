# imports
import os

import torch

# local imports
from gesonn.com3DeepShape import bernoulli_one_optimizer as bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is runCom3DeepShapeGeometry.py")

if __name__ == "__main__":

    # ==============================================================
    # Parameters to be modified freely by the user
    # ==============================================================

    train = True
    # train = False

    deepGeoDict = {
        "pde_learning_rate": 5e-3,
        "sympnet_learning_rate": 5e-3,
        "layer_sizes": [2, 40, 80, 40, 1],
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_min": 0.5,
        "rho_max": 1,
        "file_name": "bernoulli_SIAM",
        "to_be_trained": True,
        "boundary_condition": "bernoulli",
        "a": 0.6,
        "pinn_activation": torch.tanh,
    }

    epochs = 200
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
                    "./../outputs/deepShape/net/" + deepGeoDict["file_name"] + ".pth"
                )
            except FileNotFoundError:
                pass

        network = bernoulli.Geo_Net(deepGeoDict=deepGeoDict)

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
        network = bernoulli.Geo_Net(deepGeoDict=deepGeoDict)
        network.plot_result(save_plots)
