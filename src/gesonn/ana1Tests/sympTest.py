# imports
import os
import torch

# local imports
from gesonn.out1Plot import makePlots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

try:
    import torchinfo

    no_torchinfo = False
except ModuleNotFoundError:
    no_torchinfo = True

# local imports
from gesonn.com2SympNets import G

def main_symp_test(testsDict):
    for simu_name in testsDict.keys():
        simuDict = testsDict[simu_name]

        # Chargement de la solution PINNs
        simuPath = "./../outputs/SympNets/net/" + simuDict["file_name"] + ".pth"
        # Chargement du PINNs
        if not os.path.isfile(simuPath):
            print(f"Empty file for simulation {simu_name}. Computation launched")
            network = G.Symp_Net(SympNetsDict=simuDict)
            if device.type == "cpu":
                tps = network.train(
                    epochs=10_000, n_collocation=10_000, plot_history=False
                )
            else:
                tps = network.train(epochs=10_000, n_collocation=250_000, plot_history=False)
            print(f"Computational time: {str(tps)[:4]} sec.")
        else:
            network = G.Symp_Net(SympNetsDict=simuDict)

        network.plot_result(True)
        hausdorff_error = network.get_hausdorff_error()
        print("Hausdorff distance: ", hausdorff_error)
