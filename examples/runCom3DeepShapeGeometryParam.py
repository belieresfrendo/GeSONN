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

    deepGeoDict = {
        "pde_learning_rate": 1e-2,
        "sympnet_learning_rate": 1e-2,
        "layer_sizes": [3, 10, 20, 10, 1],
        "nb_of_networks": 4,
        "networks_size": 10,
        "rho_min": 0,
        "rho_max": 1,
        "mu_min": 0.5,
        "mu_max": 2,
        "file_name": "test",
        "to_be_trained": True,
        "source_term": "exp",
        "boundary_condition": "dirichlet_homogene",
    }

    epochs = 1
    n_collocation = 100_000
    new_training = False
    # new_training = True

    if train:
        if new_training:
            try:
                os.remove("./../outputs/deepShape/net/" + deepGeoDict["file_name"] + ".pth")
            except FileNotFoundError:
                pass
                
        network = geometryParam.Geo_Net(deepGeoDict=deepGeoDict)


        if device.type == "cpu":
            tps = network.train(epochs=epochs, n_collocation=n_collocation, plot_history=True)
        else:
            tps = network.train(epochs=epochs, n_collocation=n_collocation, plot_history=True)
        print(f"Computational time: {str(tps)[:4]} sec.")

    else:
        network = geometryParam.Geo_Net(deepGeoDict=deepGeoDict)
