# imports
import os

import torch

# local imports
from gesonn.com3DeepShape import geometryParam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is runCom3DeepShapeGeometryParam.py")

if __name__ == "__main__":
    train = True
    # train = False

    # ==============================================================
    # Parameters to be modified freely by the user
    # ==============================================================

    deepGeoDict = {
        "pde_learning_rate": 5e-3,
        "sympnet_learning_rate": 5e-3,
        "layer_sizes": [3, 20, 40, 40, 20, 1],
        "nb_of_networks": 6,
        "networks_size": 8,
        "rho_min": 0,
        "rho_max": 1,
        "mu_min": 0.5,
        "mu_max": 2.0,
        "file_name": "SIAM_bizaroid",
        "to_be_trained": True,
        "source_term": "bizaroid",
        "boundary_condition": "homogeneous_dirichlet",
        "pinn_activation": torch.tanh,
        "sympnet_activation": torch.tanh,
    }

    epochs = 1_000
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
                    "./../outputs/deepShape/net/param_"
                    + deepGeoDict["file_name"]
                    + ".pth"
                )
            except FileNotFoundError:
                pass

        network = geometryParam.Geo_Net(deepGeoDict=deepGeoDict)

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
        network = geometryParam.Geo_Net(deepGeoDict=deepGeoDict)
        network.plot_result(save_plots)
