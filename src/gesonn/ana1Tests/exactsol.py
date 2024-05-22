# imports
import os
import torch
import scipy.spatial.distance as dist
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

# local imports
from gesonn.ana1Tests import optimalShapes
from gesonn.com3DeepShape import geometry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")


def main_exactsol(simuDict):

    # Chargement de la solution GeSONN
    simuPath = "./../outputs/deepShape/net/" + simuDict["file_name"] + ".pth"
    # Chargement du réseau
    if not os.path.isfile(simuPath):
        print(f"Empty file for simulation. Computation launched")
        network = geometry.Geo_Net(deepGeoDict=simuDict)
        if device.type == "cpu":
            tps = network.train(epochs=10_000, n_collocation=10_000, plot_history=False)
        else:
            tps = network.train(
                epochs=25_000, n_collocation=250_000, plot_history=False
            )
        print(f"Computational time: {str(tps)[:4]} sec.")
    else:
        network = geometry.Geo_Net(deepGeoDict=simuDict)

    # ERREUR EDP
    def get_exact_sol(x, y):
        return 0.25 * (1 - x**2 - y**2) * (x**2 + y**2 < 1)

    def get_net_sol(x, y, xG, yG):
        n_pts = x.size()[0]
        x_0, y_0 = optimalShapes.translate_to_zero(x, y, n_pts)
        return network.get_u(
            x + xG * torch.ones_like(x), y + yG * torch.ones_like(y)
        ) * (x_0**2 + y_0**2 < 1)

    def get_error(x, y, xG, yG):
        xT, yT = network.apply_symplecto(x, y)
        # n_pts = xT.size()[0]
        # xT, yT = optimalShapes.translate_to_zero(xT, yT, n_pts)
        u_ex = get_exact_sol(xT, yT)
        u_pred = get_net_sol(x, y, xG, yG)
        return u_ex - u_pred

    # DISTANCE DE HAUSDORFF
    n_pts = 10_000
    shape = (n_pts, 1)
    rho = network.rho_max
    theta = network.random(
        network.theta_min, network.theta_max, shape, requires_grad=True
    )
    x_border_exact = rho * torch.cos(theta)
    y_border_exact = rho * torch.sin(theta)
    x_border_net, y_border_net = network.apply_symplecto(x_border_exact, y_border_exact)

    xG, yG = x_border_net.sum() / n_pts, y_border_net.sum() / n_pts

    x_border_net, y_border_net = optimalShapes.translate_to_zero(
        x_border_net, y_border_net, n_pts
    )
    XT_net = []
    X_ex = []
    for x, y in zip(x_border_net.flatten().tolist(), y_border_net.flatten().tolist()):
        XT_net.append((x, y))
    for x, y in zip(
        x_border_exact.flatten().tolist(), y_border_exact.flatten().tolist()
    ):
        X_ex.append((x, y))

    XT_net = np.array(XT_net)
    X_ex = np.array(X_ex)

    hausdorff_error = max(
        dist.directed_hausdorff(XT_net, X_ex)[0],
        dist.directed_hausdorff(X_ex, XT_net)[0],
    )

    xT_min, xT_max = x_border_net.min().item(), x_border_net.max().item()
    yT_min, yT_max = y_border_net.min().item(), y_border_net.max().item()
    lx = xT_max - xT_min
    ly = yT_max - yT_min
    xT_max += 0.025 * max(lx, ly)
    xT_min -= 0.025 * max(lx, ly)
    yT_max += 0.025 * max(lx, ly)
    yT_min -= 0.025 * max(lx, ly)

    _, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    ax.scatter(
        x_border_exact.detach().cpu(),
        y_border_exact.detach().cpu(),
        s=10,
        c="red",
        label="exact solution",
    )
    ax.scatter(
        x_border_net.detach().cpu(),
        y_border_net.detach().cpu(),
        s=1,
        c="green",
        label="GeSONN solution",
    )
    ax.set_aspect("equal")
    plt.savefig("./../outputs/deepShape/img/shape_" + simuDict["file_name"] + ".pdf")
    plt.plot()

    n_visu = 768
    n_contour = 250
    draw_contours = True
    n_drawn_contours = 10

    # make meshgrid
    x = torch.linspace(
        xT_min, xT_max, n_visu, dtype=torch.float64, device=device, requires_grad=True
    )
    x_ = torch.tile(x, (n_visu,))[:, None]
    y = torch.linspace(
        yT_min, yT_max, n_visu, dtype=torch.float64, device=device, requires_grad=True
    )
    y_ = torch.repeat_interleave(y, n_visu)[:, None]

    # apply the inverse symplecto to get u
    xT_inv, yT_inv = network.apply_inverse_symplecto(x_, y_)
    # u_net = network.get_u(xT_inv, yT_inv)[:, 0].detach().cpu().reshape((n_visu, n_visu))
    # u = network.get_u(xT_inv, yT_inv)[:, 0].detach().cpu().reshape((n_visu, n_visu))
    err = get_error(xT_inv, yT_inv, xG, yG)[:, 0].detach().cpu().reshape((n_visu, n_visu))
    u_ex = get_exact_sol(x_, y_)[:, 0].detach().cpu().reshape((n_visu, n_visu))
    u_net = get_net_sol(xT_inv, yT_inv, xG, yG)[:, 0].detach().cpu().reshape((n_visu, n_visu))

    # mask u outside the domain
    x, y = np.meshgrid(x.detach().cpu(), y.detach().cpu())
    xT_inv, yT_inv = xT_inv.detach().cpu(), yT_inv.detach().cpu()
    # mask = (xT_inv**2 + yT_inv**2 > network.rho_max**2) | (xT_inv**2 + yT_inv**2 < network.rho_min**2)
    # u_net = np.ma.array(u_net, mask=mask)

    # draw the contours
    fig, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    im = ax.contourf(
        x,
        y,
        err,
        n_contour,
        cmap="gist_ncar",
        zorder=-9,
    )

    if draw_contours:
        ax.contour(
            im,
            levels=im.levels[:: n_contour // n_drawn_contours],
            zorder=-9,
            colors="white",
            alpha=0.5,
            linewidths=0.8,
        )
    fig.colorbar(im, ax=ax)

    ax.set_aspect("equal")
    plt.gca().set_rasterization_zorder(-1)
    plt.savefig("./../outputs/deepShape/img/error_" + simuDict["file_name"] + ".pdf")
    plt.show()

    # draw the contours
    fig, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    im = ax.contourf(
        x,
        y,
        u_ex,
        n_contour,
        cmap="gist_ncar",
        zorder=-9,
    )

    if draw_contours:
        ax.contour(
            im,
            levels=im.levels[:: n_contour // n_drawn_contours],
            zorder=-9,
            colors="white",
            alpha=0.5,
            linewidths=0.8,
        )
    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    plt.gca().set_rasterization_zorder(-1)
    plt.savefig("./../outputs/deepShape/img/exact_" + simuDict["file_name"] + ".pdf")
    plt.show()

    # draw the contours
    fig, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    im = ax.contourf(
        x,
        y,
        u_net,
        n_contour,
        cmap="gist_ncar",
        zorder=-9,
    )
    ax.set_aspect("equal")

    if draw_contours:
        ax.contour(
            im,
            levels=im.levels[:: n_contour // n_drawn_contours],
            zorder=-9,
            colors="white",
            alpha=0.5,
            linewidths=0.8,
        )
    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    plt.gca().set_rasterization_zorder(-1)
    plt.savefig("./../outputs/deepShape/img/net_" + simuDict["file_name"] + ".pdf")
    plt.show()

    n_border = 10_000
    theta = torch.linspace(0, 2*torch.pi, n_border, requires_grad=True)[:, None]
    x_border, y_border = network.rho_max * torch.cos(theta), network.rho_max * torch.sin(theta)
    xT_border, yT_border = network.apply_symplecto(x_border, y_border)
    dn_u = network.get_dn_u(x_border, y_border)
    fig, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))
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
    plt.savefig("./../outputs/deepShape/img/optimality_cond_" + simuDict["file_name"] + ".pdf")


    print(f"Hausdorff distance: {hausdorff_error:3.2e}")
    print(f"L2 error: {math.sqrt((err**2).sum()/(n_visu**2) * (xT_max - xT_min) * (yT_max - yT_min)):3.2e}")
