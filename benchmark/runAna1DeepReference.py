# imports
import os
import torch

# local imports
from gesonn.ana1Tests import deepReference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is runCom3DeepShapeGeometry.py")

if __name__ == "__main__":
    train = True
    # train = False

    circleDict = {
        "pde_learning_rate": 1e-2,
        "sympnet_learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 10, 1],
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "benchmark_ff_circle",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "dirichlet_homogene",
    }

    expDict = {
        "pde_learning_rate": 1e-2,
        "sympnet_learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 10, 1],
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "benchmark_ff_exp",
        "to_be_trained": True,
        "source_term": "exp",
        "boundary_condition": "dirichlet_homogene",
    }

    testsDict = {
        "circle": circleDict,
        "exp": expDict,
    }

    deepReference.main_reference_test(testsDict)