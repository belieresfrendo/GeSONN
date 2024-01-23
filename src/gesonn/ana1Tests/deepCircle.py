# imports
import os
import torch

# local imports
from gesonn.com3DeepShape import geometry
from gesonn.ana1Tests import optimalShapes
from gesonn.out1Plot import makePlots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is runCom3DeepShapeGeometry.py")

def main_deep_circle_test(simuDict):

    # Chargement de la solution PINNs
    simuPath = "./../outputs/deepShape/net/" + simuDict["file_name"] + ".pth"
    # Chargement du PINNs
    if not os.path.isfile(simuPath):
        print(f"Empty file for simulation circle. Computation launched")
        network = geometry.Geo_Net(deepGeoDict=simuDict)
        if device.type == "cpu":
            network.train(
                epochs=1_000, n_collocation=10_000, plot_history=False
            )
        else:
            network.train(epochs=10_000, n_collocation=250_000, plot_history=False)
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
    x_ex = rho * torch.cos(theta)
    y_ex = rho * torch.sin(theta)

    x_net, y_net = network.apply_symplecto(x_ex, y_ex)
    x_net, y_net = optimalShapes.translate_to_zero(x_net, y_net, n_pts, network.Vol)
    X_net = []
    X_ex = []
    for x, y in zip(x_net.flatten().tolist(), y_net.flatten().tolist()):
        X_net.append((x, y))
    for x, y in zip(x_ex.flatten().tolist(), y_ex.flatten().tolist()):
        X_ex.append((x, y))

    X_net = np.array(X_net)
    X_ex = np.array(X_ex)

    hausdorff_error = max(dist.directed_hausdorff(X_net, X_ex)[0], dist.directed_hausdorff(X_ex, X_net)[0])
    print("Dictance de Hausdorff:", hausdorff_error)
    makePlots.shape_error(
        x_net.detach().cpu(),
        y_net.detach().cpu(),
        x_ex.detach().cpu(),
        y_ex.detach().cpu(),
        title=f"Hausdorff error: {hausdorff_error:5.2e}"
    )
