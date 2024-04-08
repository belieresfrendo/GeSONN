# imports
import os
import torch
import pandas as pd

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
from gesonn.com1PINNs import poisson
from gesonn.com1PINNs import metricTensors


def main_poisson_test(testsDict, source_term):
    for simu_name in testsDict.keys():
        simuDict = testsDict[simu_name]

        # Chargement de la solution FreeFem++
        fem_path = "./exact_solutions/" + simu_name + "_source_" + source_term + ".csv"
        if os.path.isfile(fem_path):
            dict = pd.read_csv(fem_path, delimiter=";")
        else:
            raise FileNotFoundError("Could not find fem solution storage file")

        X = torch.tensor(dict["x"], requires_grad=True, device=device)[:, None]
        Y = torch.tensor(dict["y"], requires_grad=True, device=device)[:, None]
        Ufem = torch.tensor(dict["u"], requires_grad=True, device=device)[:, None]

        # Chargement de la solution PINNs
        simuPath = "./../outputs/PINNs/net/" + simuDict["file_name"] + ".pth"
        # Chargement du PINNs
        if not os.path.isfile(simuPath):
            print(f"Empty file for simulation {simu_name}. Computation launched")
            network = poisson.PINNs(PINNsDict=simuDict)
            if device.type == "cpu":
                tps = network.train(
                    epochs=10_000, n_collocation=10_000, plot_history=False
                )
            else:
                tps = network.train(
                    epochs=10_000, n_collocation=250_000, plot_history=False
                )
            print(f"Computational time: {str(tps)[:4]} sec.")
        else:
            network = poisson.PINNs(PINNsDict=simuDict)

        X_visu, Y_visu = X, Y
        if simuDict["symplecto_name"] is not None:
            X, Y = metricTensors.apply_symplecto(X, Y, name="inverse_ellipse_benchmark")
        Unet = network.get_u(X, Y)

        X_visu = X_visu.detach().cpu()
        Y_visu = Y_visu.detach().cpu()

        Unet = (
            (
                Unet
                * (X**2 + Y**2 <= network.rho_max**2)
                * (X**2 + Y**2 >= network.rho_min**2)
            )
            .detach()
            .cpu()
        )
        Ufem = Ufem.detach().cpu()
        err = abs(Unet - Ufem)
        errL2 = ((Unet - Ufem) ** 2).sum() / X.size()[0] * network.Vol

        print("erreur L2 : ", errL2.item())

        # Affichage
        fig_path = "./../outputs/PINNs/img/" + simuDict["file_name"]
        makePlots.edp(X_visu, Y_visu, Unet, True, fig_path + "_PINNS", "PINNs")
        makePlots.edp(X_visu, Y_visu, Ufem, True, fig_path + "_FEM", "FEM")
        makePlots.edp(X_visu, Y_visu, err, True, fig_path + "_ERR", "error")
