# imports
import os

import torch

# local imports
from gesonn.com3DeepShape import geometryParam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is runCom3DeepShapeGeometry.py")

if __name__ == "__main__":
    train = True
    # train = False

    # ==============================================================
    # Parameters to be modified freely by the user
    # ==============================================================

    deepGeoDict = {
        "pde_learning_rate": 5e-4,
        "sympnet_learning_rate": 5e-4,
        "layer_sizes": [3, 10, 20, 80, 20, 10, 1],
        "nb_of_networks": 4,
        "networks_size": 8,
        "rho_min": 0,
        "rho_max": 1,
        "mu_min": 0.5,
        "mu_max": 1.5,
        "file_name": "default",
        "to_be_trained": True,
        "source_term": "ellipse",
        "boundary_condition": "homogeneous_dirichlet",
    }

    epochs = 5
    n_collocation = 10_000
    # new_training = False
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
                    "./../outputs/deepShape/net/param_" + deepGeoDict["file_name"] + ".pth"
                )
                print("bite")
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
