# imports
import os
import torch
import pandas as pd
import math

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
        a_fem_path = "./shape_references/ff_" + simu_name + ".csv"
        u_fem_path = "./optimal_edp/ff_edp_" + simu_name + ".csv"
        if os.path.isfile(a_fem_path):
            a_dict = pd.read_csv(a_fem_path, delimiter=";")
        else:
            raise FileNotFoundError("Could not find fem solution storage file")
        if os.path.isfile(u_fem_path):
            u_dict = pd.read_csv(u_fem_path, delimiter=";")
        else:
            raise FileNotFoundError("Could not find fem solution storage file")

        X = a_dict["x"]
        Y = a_dict["y"]
        A_fem = a_dict["a"]
        U_fem = u_dict["u"]

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
                tps = network.train(
                    epochs=25_000, n_collocation=250_000, plot_history=False
                )
            print(f"Computational time: {str(tps)[:4]} sec.")
        else:
            network = geometry.Geo_Net(deepGeoDict=simuDict)

        n_pts = 10_000
        shape = (n_pts, 1)
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
        if simuDict["source_term"] == "one":
            xT_net, yT_net = optimalShapes.translate_to_zero(xT_net, yT_net, n_pts)
            xT_map, yT_map = optimalShapes.translate_to_zero(xT_map, yT_map, 50_000)
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
            if border_bool == 1 and A_fem[i] == 1 and A_fem[i + 1] == 0:
                x, y = X[i], Y[i]
                X_fem.append((x, y))
                x_fem.append(x)
                y_fem.append(y)
            cpt = cpt + 1
            if cpt == step:
                cpt = 0
                border_bool = 0

        XT_net = np.array(XT_net)
        X_fem = np.array(X_fem)

        hausdorff_error = max(
            dist.directed_hausdorff(XT_net, X_fem)[0],
            dist.directed_hausdorff(X_fem, XT_net)[0],
        )
        print("Dictance de Hausdorff:", hausdorff_error)
        makePlots.deep_shape_error(
            network.rho_max,
            lambda x, y: network.apply_symplecto(x, y),
            x_fem,
            y_fem,
            True,
            "./../outputs/deepShape/img/" + simuDict["file_name"] + ".pdf",
        )
        makePlots.edp_shape_error(
            u_pred.detach().cpu(),
            xT_map.detach().cpu(),
            yT_map.detach().cpu(),
            x_fem,
            y_fem,
            True,
            "./../outputs/deepShape/img/" + simuDict["file_name"] + ".pdf",
        )

        # ==================================================================
        #   EDP
        # ==================================================================


        xT, yT = torch.tensor(X)[:, None], torch.tensor(Y)[:, None]
        x, y = network.apply_inverse_symplecto(xT, yT)
        cond = x**2 + y**2 <= network.rho_max**2
        x, y, xT, yT = x[cond][:, None], y[cond][:, None], xT[cond][:, None], yT[cond][:, None]

        xT_min, xT_max = xT.min().item(), xT.max().item()
        yT_min, yT_max = yT.min().item(), yT.max().item()
        lx = xT_max - xT_min
        ly = yT_max - yT_min
        xT_max += 0.025 * max(lx, ly)
        xT_min -= 0.025 * max(lx, ly)
        yT_max += 0.025 * max(lx, ly)
        yT_min -= 0.025 * max(lx, ly)

        u_pred = network.get_u(x, y)
        u_fem = torch.tensor(U_fem)[:, None][cond][:, None]
        err = (u_pred - u_fem) * (x**2 + y**2 < network.rho_max**2)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5 * lx / ly, 5 * ly / lx))

        im = ax.scatter(
            xT.detach().cpu(),
            yT.detach().cpu(),
            s=5,
            c=u_fem.detach().cpu(),
            cmap="gist_ncar",
            label="ff++",
        )
        fig.colorbar(im, ax=ax)
        ax.set_aspect("equal")
        plt.savefig("./../outputs/deepShape/img/EDP_FF" + simuDict["file_name"] + ".pdf")


        fig, ax = plt.subplots(1, 1, figsize=(5 * lx / ly, 5 * ly / lx))
        im = ax.scatter(
            xT.detach().cpu(),
            yT.detach().cpu(),
            s=5,
            c=u_pred.detach().cpu(),
            cmap="gist_ncar",
            label="gesonn",
        )
        fig.colorbar(im, ax=ax)
        ax.set_aspect("equal")
        plt.savefig("./../outputs/deepShape/img/EDP_GESONN" + simuDict["file_name"] + ".pdf")


        fig, ax = plt.subplots(1, 1, figsize=(5 * lx / ly, 5 * ly / lx))
        im = ax.scatter(
            xT.detach().cpu(),
            yT.detach().cpu(),
            s=5,
            c=err.detach().cpu(),
            cmap="gist_ncar",
            label="error",
        )
        fig.colorbar(im, ax=ax)
        ax.set_aspect("equal")
        plt.savefig("./../outputs/deepShape/img/ERROR_GESONN_EDP" + simuDict["file_name"] + ".pdf")


        n_border = 10_000
        theta = torch.linspace(0, 2*torch.pi, n_border, requires_grad=True)[:, None]
        x_border, y_border = network.rho_max * torch.cos(theta), network.rho_max * torch.sin(theta)
        xT_border, yT_border = network.apply_symplecto(x_border, y_border)
        dn_u = network.get_dn_u(x_border, y_border)
        fig, ax = plt.subplots(1, 1, figsize=(5 * lx / ly, 5 * ly / lx))
        im = ax.scatter(
            xT_border.detach().cpu(),
            yT_border.detach().cpu(),
            s=1,
            c=dn_u.detach().cpu(),
            cmap="gist_ncar",
            label="error",
        )
        fig.colorbar(im, ax=ax)
        ax.set_aspect("equal")
        plt.savefig("./../outputs/deepShape/img/OPTIMALITY_CONDITION_GESONN_" + simuDict["file_name"] + ".pdf")

        plt.show()

    print(f"Hausdorff distance: {hausdorff_error:3.2e}")
    print(f"L2 error: {math.sqrt((err**2).sum()/(xT.size()[0]) * (xT_max - xT_min) * (yT_max - yT_min)):3.2e}")
