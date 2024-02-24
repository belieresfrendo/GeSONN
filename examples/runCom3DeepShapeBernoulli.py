# imports
import os

import torch

# local imports
from gesonn.com3DeepShape import bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is runCom3DeepShapeGeometry.py")

if __name__ == "__main__":

    #==============================================================
    # Parameters to be modified freely by the user
    #==============================================================

    train = True
    # train = False

    deepGeoDict = {
        "pde_learning_rate": 1e-2,
        "sympnet_learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 10, 1],
        "nb_of_networks": 7,
        "networks_size": 5,
        "rho_min": 0.5,
        "rho_max": 1,
        "file_name": "bernoulli",
        "to_be_trained": True,
        "boundary_condition": "c moua",
        "a": 0.8,
    }

    epochs = 10_000
    n_collocation = 10_000
    new_training = False
    # new_training = True

    #==============================================================
    # End of the modifiable area
    #==============================================================

    if train:
        if new_training:
            try:
                os.remove("./../outputs/deepShape/net/" + deepGeoDict["file_name"] + ".pth")
            except FileNotFoundError:
                pass

        network = bernoulli.Geo_Net(deepGeoDict=deepGeoDict)


        if device.type == "cpu":
            tps = network.train(epochs=epochs, n_collocation=n_collocation, plot_history=True)
        else:
            tps = network.train(epochs=epochs, n_collocation=n_collocation, plot_history=True)
        print(f"Computational time: {str(tps)[:4]} sec.")

    else:
        network = bernoulli.Geo_Net(deepGeoDict=deepGeoDict)
