# imports
import torch

# local imports
from gesonn.ana1Tests import exactsol

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is runCom3DeepShapeGeometry.py")

if __name__ == "__main__":
    train = True
    # train = False

    circleDict = {
        "pde_learning_rate": 1e-3,
        "sympnet_learning_rate": 1e-3,
        "layer_sizes": [2, 10, 20, 80, 20, 10, 1],
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "benchmark_exact_circle",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "homogeneous_dirichlet",
    }

    exactsol.main_exactsol(circleDict)
