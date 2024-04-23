import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["fontenc"], "size": 15})
rc("text", usetex=True)


def loss(loss_history, save_plots, name):
    _, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_yscale("symlog", linthresh=1e-4)
    if save_plots:
        plt.savefig(name + "_loss.pdf")
    plt.show()


def edp(x, y, u, save_plots, name, title=None, figsize=(7.5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.scatter(
        x,
        y,
        s=1,
        c=u,
        cmap="gist_ncar",
    )
    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if save_plots:
        plt.savefig(name + ".pdf")
    plt.show()


def edp_contour(
    rho_min,
    rho_max,
    get_u,
    apply_symplecto,
    apply_inverse_symplecto,
    n_visu=768,
    n_contour=250,
    draw_contours=True,
    n_drawn_contours=10,
):
    import numpy as np
    import torch

    # measuring the min and max coordinates of the bounding box
    theta = torch.linspace(0, 2 * np.pi, 1_000, dtype=torch.float64)[:, None]
    x = rho_max * torch.cos(theta)
    y = rho_max * torch.sin(theta)
    x, y = apply_symplecto(x, y)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    lx = x_max - x_min
    ly = y_max - y_min
    x_max += 0.025 * max(lx, ly)
    x_min -= 0.025 * max(lx, ly)
    y_max += 0.025 * max(lx, ly)
    y_min -= 0.025 * max(lx, ly)

    # make meshgrid
    x = torch.linspace(x_min, x_max, n_visu, dtype=torch.float64)
    x_ = torch.tile(x, (n_visu,))[:, None]
    y = torch.linspace(y_min, y_max, n_visu, dtype=torch.float64)
    y_ = torch.repeat_interleave(y, n_visu)[:, None]

    # apply the inverse symplecto to get u
    xT_inv, yT_inv = apply_inverse_symplecto(x_, y_)
    u = get_u(xT_inv, yT_inv)[:, 0].detach().cpu().reshape((n_visu, n_visu))

    # mask u outside the domain
    x, y = np.meshgrid(x.detach().cpu(), y.detach().cpu())
    mask = (xT_inv**2 + yT_inv**2 > rho_max**2) | (xT_inv**2 + yT_inv**2 < rho_min**2)
    u = np.ma.array(u, mask=mask)

    # draw the contours
    fig, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    im = ax.contourf(
        x,
        y,
        u,
        n_contour,
        cmap="turbo",
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
    plt.show()


def shape(x, y, save_plots, name, title=None):
    _, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(
        x,
        y,
        s=1,
    )
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if save_plots:
        plt.savefig(name + ".pdf")

    plt.show()


def param_shape(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, save_plots, name, title=None):
    _, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x1, y1, s=1)
    ax.scatter(x2, y2, s=1)
    ax.scatter(x3, y3, s=1)
    ax.scatter(x4, y4, s=1)
    ax.scatter(x5, y5, s=1)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if save_plots:
        plt.savefig(name + "_superposititon.pdf")

    plt.show()


def shape_error(x, y, u, v, save_plots, name, title=None):
    _, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(
        u,
        v,
        s=50,
        c="red",
        label="fixed point optimal shape",
    )
    ax.scatter(
        x,
        y,
        s=1,
        c="green",
        label="GeSONN optimal shape",
    )
    ax.set_aspect("equal")
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_plots:
        plt.savefig(name + "_error.pdf")
    plt.show()


def edp_shape_error(edp, x, y, u, v, save_plots, name, title=None):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.scatter(
        x,
        y,
        s=1,
        c=edp,
        label="PDE GeSONN prediction",
        cmap="gist_ncar",
    )
    ax.scatter(
        u,
        v,
        s=10,
        c="red",
        label="fixed point optimal shape",
    )
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", borderaxespad=0.0, ncol=2)
    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if save_plots:
        plt.savefig(name + "_edp_error.pdf")

    plt.show()
