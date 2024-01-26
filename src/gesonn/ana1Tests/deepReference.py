# imports
import os
import torch
import pandas as pd

# local imports
from gesonn.out1Plot import makePlots
from gesonn.ana1Tests import optimalShapes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")
try:
    import torchinfo

    no_torchinfo = False
except ModuleNotFoundError:
    no_torchinfo = True

# local imports
from gesonn.com3DeepShape import geometry


def main_reference_test(testsDict):
    for simu_name in testsDict.keys():
        simuDict = testsDict[simu_name]

        # Chargement de la solution FreeFem++
        fem_path = "./shape_references/ff_" + simu_name + ".csv"
        if os.path.isfile(fem_path):
            dict = pd.read_csv(fem_path, delimiter=";")
        else:
            raise FileNotFoundError("Could not find fem solution storage file")

        X = dict["x"]
        Y = dict["y"]
        A_fem = dict["a"]

        # Chargement de la solution GeSONN
        simuPath = "./../outputs/deepShape/net/" + simuDict["file_name"] + ".pth"
        # Chargement du réseau
        if not os.path.isfile(simuPath):
            print(f"Empty file for simulation {simu_name}. Computation launched")
            network = geometry.Geo_Net(deepGeoDict=simuDict)
            if device.type == "cpu":
                tps = network.train(
                    epochs=10_000, n_collocation=10_000, plot_history=False
                )
            else:
                tps = network.train(epochs=25_000, n_collocation=100_000, plot_history=False)
            print(f"Computational time: {str(tps)[:4]} sec.")
        else:
            network = geometry.Geo_Net(deepGeoDict=simuDict)

        n_pts = 10_000
        shape = (n_pts,1)
        network.make_collocation(n_pts)

        import scipy.spatial.distance as dist
        import numpy as np

        rho = network.rho_max
        theta = network.random(
            network.theta_min, network.theta_max, shape, requires_grad=True
        )
        x_net = rho * torch.cos(theta)
        y_net = rho * torch.sin(theta)

        xT_net, yT_net = network.apply_symplecto(x_net, y_net)
        network.make_collocation(50_000)
        u_pred = network.get_u(network.x_collocation, network.y_collocation)
        xT_map, yT_map = network.apply_symplecto(
            network.x_collocation,
            network.y_collocation,
        )
        if simuDict["source_term"]=="one":
            xT_net, yT_net = optimalShapes.translate_to_zero(xT_net, yT_net, n_pts, network.Vol)
            xT_map, yT_map = optimalShapes.translate_to_zero(xT_map, yT_map, 50_000, network.Vol)
        XT_net = []
        X_fem = []
        x_fem, y_fem = [], []
        for x, y in zip(xT_net.flatten().tolist(), yT_net.flatten().tolist()):
            XT_net.append((x, y))
        step = 200
        cpt = 0
        border_bool = 0
        for i in range(len(A_fem)):
            if border_bool == 0 and A_fem[i] == 1:
                border_bool = 1
                x, y = X[i], Y[i]
                X_fem.append((x, y))
                x_fem.append(x)
                y_fem.append(y)
            if border_bool == 1 and A_fem[i] == 1 and A_fem[i+1] == 0:
                x, y = X[i], Y[i]
                X_fem.append((x, y))
                x_fem.append(x)
                y_fem.append(y)
            cpt = cpt + 1
            if cpt==step:
                cpt = 0
                border_bool = 0



        XT_net = np.array(XT_net)
        X_fem = np.array(X_fem)

        hausdorff_error = max(dist.directed_hausdorff(XT_net, X_fem)[0], dist.directed_hausdorff(X_fem, XT_net)[0])
        print("Dictance de Hausdorff:", hausdorff_error)
        makePlots.shape_error(
            xT_net.detach().cpu(),
            yT_net.detach().cpu(),
            x_fem,
            y_fem,
            title=f"Hausdorff error: {hausdorff_error:5.2e}"
        )
        makePlots.edp_shape_error(
            u_pred.detach().cpu(),
            xT_map.detach().cpu(),
            yT_map.detach().cpu(),
            x_fem,
            y_fem,
            title=f"Hausdorff error: {hausdorff_error:5.2e}"
        )
