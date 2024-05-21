import matplotlib.pyplot as plt
import torch
from gesonn.ana1Tests.optimalShapes import translate_to_zero
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["fontenc"], "size": 15})
rc("text", usetex=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")


def loss(loss_history, save_plots, name):
    _, ax = plt.subplots()
    ax.plot(loss_history)
    history = torch.tensor(loss_history)
    ax.set_yscale("symlog", linthresh=abs(history).min().item())
    if save_plots:
        plt.savefig(name + "_loss.pdf")
    plt.show()


def edp_contour(
    rho_min,
    rho_max,
    get_u,
    apply_symplecto,
    apply_inverse_symplecto,
    save_plots,
    name,
    n_visu=768,
    n_contour=250,
    draw_contours=True,
    n_drawn_contours=10,
):
    import numpy as np
    import torch

    # measuring the min and max coordinates of the bounding box
    theta = torch.linspace(
        0, 2 * np.pi, 10_000, dtype=torch.float64, requires_grad=True
    )[:, None]
    x = rho_max * torch.cos(theta)
    y = rho_max * torch.sin(theta)
    x, y = apply_symplecto(x, y)
    x_min, x_max = x.min().item(), x.max().item()
    y_min, y_max = y.min().item(), y.max().item()
    lx = x_max - x_min
    ly = y_max - y_min
    x_max += 0.025 * max(lx, ly)
    x_min -= 0.025 * max(lx, ly)
    y_max += 0.025 * max(lx, ly)
    y_min -= 0.025 * max(lx, ly)

    # make meshgrid
    x = torch.linspace(
        x_min, x_max, n_visu, dtype=torch.float64, device=device, requires_grad=True
    )
    x_ = torch.tile(x, (n_visu,))[:, None]
    y = torch.linspace(
        y_min, y_max, n_visu, dtype=torch.float64, device=device, requires_grad=True
    )
    y_ = torch.repeat_interleave(y, n_visu)[:, None]

    # apply the inverse symplecto to get u
    xT_inv, yT_inv = apply_inverse_symplecto(x_, y_)
    u = get_u(xT_inv, yT_inv)[:, 0].detach().cpu().reshape((n_visu, n_visu))

    # mask u outside the domain
    x, y = np.meshgrid(x.detach().cpu(), y.detach().cpu())
    xT_inv, yT_inv = xT_inv.detach().cpu(), yT_inv.detach().cpu()
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
    if save_plots:
        plt.savefig(name + ".pdf")
    plt.show()


def edp_contour_param(
    rho_min,
    rho_max,
    mu_min,
    mu_max,
    get_u,
    apply_symplecto,
    apply_inverse_symplecto,
    save_plots,
    name,
    n_visu=768,
    n_contour=250,
    draw_contours=True,
    n_drawn_contours=10,
):
    import numpy as np
    import torch

    mu_list = [
        mu_min,
        0.75 * mu_min + 0.25 * mu_max,
        0.5 * mu_min + 0.5 * mu_max,
        0.25 * mu_min + 0.75 * mu_max,
        mu_max,
    ]

    for mu in mu_list:
        # measuring the min and max coordinates of the bounding box
        theta = torch.linspace(0, 2 * np.pi, 10_000, dtype=torch.float64)[:, None]
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
        x = torch.linspace(
            x_min, x_max, n_visu, dtype=torch.float64, device=device, requires_grad=True
        )
        x_ = torch.tile(x, (n_visu,))[:, None]
        y = torch.linspace(
            y_min, y_max, n_visu, dtype=torch.float64, device=device, requires_grad=True
        )
        y_ = torch.repeat_interleave(y, n_visu)[:, None]
        ones_ = torch.ones_like(x_)
        mu_visu_ = mu * ones_

        # apply the inverse symplecto to get u
        xT_inv, yT_inv = apply_inverse_symplecto(x_, y_)
        u = (
            get_u(xT_inv, yT_inv, mu_visu_)[:, 0]
            .detach()
            .cpu()
            .reshape((n_visu, n_visu))
        )

        # mask u outside the domain
        x, y = np.meshgrid(x.detach().cpu(), y.detach().cpu())
        xT_inv, yT_inv = xT_inv.detach().cpu(), yT_inv.detach().cpu()
        mask = (xT_inv**2 + yT_inv**2 > rho_max**2) | (
            xT_inv**2 + yT_inv**2 < rho_min**2
        )
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
        if save_plots:
            plt.savefig(name + f"_mu_{mu:3.2f}.pdf")
        plt.show()


def edp_contour_param_source(
    rho_min,
    rho_max,
    mu_min,
    mu_max,
    get_u,
    apply_symplecto,
    apply_inverse_symplecto,
    save_plots,
    name,
    n_visu=768,
    n_contour=250,
    draw_contours=True,
    n_drawn_contours=10,
):
    import numpy as np
    import torch

    mu_list = [
        mu_min,
        0.75 * mu_min + 0.25 * mu_max,
        0.5 * mu_min + 0.5 * mu_max,
        0.25 * mu_min + 0.75 * mu_max,
        mu_max,
    ]

    for mu in mu_list:
        # measuring the min and max coordinates of the bounding box
        theta = torch.linspace(0, 2 * np.pi, 10_000, dtype=torch.float64)[:, None]
        x = rho_max * torch.cos(theta)
        y = rho_max * torch.sin(theta)
        x, y = apply_symplecto(x, y, mu * torch.ones_like(x))
        x_min, x_max = x.min().item(), x.max().item()
        y_min, y_max = y.min().item(), y.max().item()
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
        ones_ = torch.ones_like(x_)
        mu_visu_ = mu * ones_

        # apply the inverse symplecto to get u
        xT_inv, yT_inv = apply_inverse_symplecto(x_, y_, mu_visu_)
        u = (
            get_u(xT_inv, yT_inv, mu_visu_)[:, 0]
            .detach()
            .cpu()
            .reshape((n_visu, n_visu))
        )
        # mask u outside the domain
        x, y = np.meshgrid(x.detach().cpu(), y.detach().cpu())

        mask = (xT_inv**2 + yT_inv**2 > rho_max**2) | (
            xT_inv**2 + yT_inv**2 < rho_min**2
        )
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
        if save_plots:
            plt.savefig(name + f"_mu_{mu:3.2f}.pdf")
        plt.show()


def edp_contour_bernoulli(
    rho_max,
    a,
    b,
    get_u,
    apply_symplecto,
    apply_inverse_symplecto,
    save_plots,
    name,
    n_visu=768,
    n_contour=250,
    draw_contours=True,
    n_drawn_contours=10,
):
    import numpy as np
    import torch

    # measuring the min and max coordinates of the bounding box
    theta = torch.linspace(0, 2 * np.pi, 10_000, dtype=torch.float64, device=device)[
        :, None
    ]
    x = rho_max * torch.cos(theta)
    y = rho_max * torch.sin(theta)
    x, y = apply_symplecto(x, y)
    x_min, x_max = x.min().item(), x.max().item()
    y_min, y_max = y.min().item(), y.max().item()
    lx = x_max - x_min
    ly = y_max - y_min
    x_max += 0.025 * max(lx, ly)
    x_min -= 0.025 * max(lx, ly)
    y_max += 0.025 * max(lx, ly)
    y_min -= 0.025 * max(lx, ly)

    # make meshgrid
    x = torch.linspace(x_min, x_max, n_visu, device=device, dtype=torch.float64)
    x_ = torch.tile(x, (n_visu,))[:, None]
    y = torch.linspace(y_min, y_max, n_visu, device=device, dtype=torch.float64)
    y_ = torch.repeat_interleave(y, n_visu)[:, None]

    # apply the inverse symplecto to get u
    xT_inv, yT_inv = apply_inverse_symplecto(x_, y_)
    u = get_u(xT_inv, yT_inv)[:, 0].detach().cpu().reshape((n_visu, n_visu))

    # mask u outside the domain
    x, y = np.meshgrid(x.detach().cpu(), y.detach().cpu())
    xT_inv, yT_inv = xT_inv.detach().cpu(), yT_inv.detach().cpu()
    x_, y_ = x_.detach().cpu(), y_.detach().cpu()
    mask = (xT_inv**2 + yT_inv**2 > rho_max**2) | ((x_ / a) ** 2 + (y_ / b) ** 2 < 1)
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
    if save_plots:
        plt.savefig(name + ".pdf")
    plt.show()


def shape(rho_max, apply_symplecto, save_plots, name):
    import numpy as np
    import torch

    # measuring the min and max coordinates of the bounding box
    theta = torch.linspace(0, 2 * np.pi, 10_000, dtype=torch.float64)[:, None]
    x = rho_max * torch.cos(theta)
    y = rho_max * torch.sin(theta)
    xT, yT = apply_symplecto(x, y)
    xT_min, xT_max = xT.min().item(), xT.max().item()
    yT_min, yT_max = yT.min().item(), yT.max().item()
    lx = xT_max - xT_min
    ly = yT_max - yT_min

    # draw the contours
    _, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    ax.scatter(
        xT.detach().cpu(),
        yT.detach().cpu(),
        s=1,
    )
    ax.set_aspect("equal")
    if save_plots:
        plt.savefig(name + ".pdf")

    plt.show()


def shape_error(
    rho_max,
    apply_symplecto,
    apply_exact_symplecto,
    get_hausdorff_distance,
    save_plots,
    name,
):
    import numpy as np
    import torch

    # measuring the min and max coordinates of the bounding box
    theta = torch.linspace(0, 2 * np.pi, 10_000, dtype=torch.float64)[:, None]
    x = rho_max * torch.cos(theta)
    y = rho_max * torch.sin(theta)
    xT_pred, yT_pred = apply_symplecto(x, y)
    xT_ex, yT_ex = apply_exact_symplecto(x, y)
    xT_min, xT_max = xT_pred.min().item(), xT_pred.max().item()
    yT_min, yT_max = yT_pred.min().item(), yT_pred.max().item()
    lx = xT_max - xT_min
    ly = yT_max - yT_min

    # draw the contours
    _, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    ax.scatter(
        xT_ex.detach().cpu(),
        yT_ex.detach().cpu(),
        s=25,
        c="red",
    )
    ax.scatter(
        xT_pred.detach().cpu(),
        yT_pred.detach().cpu(),
        s=1,
        c="green",
    )

    if save_plots:
        plt.savefig(name + "_error.pdf")
    plt.show()

    print(f"Haussdorf distance: {get_hausdorff_distance():3.2e}")


def param_shape_error(
    rho_max,
    mu_min,
    mu_max,
    apply_symplecto,
    apply_exact_symplecto,
    get_hausdorff_distance,
    save_plots,
    name,
):
    import numpy as np
    import torch

    mu_list = [
        mu_min,
        0.75 * mu_min + 0.25 * mu_max,
        0.5 * mu_min + 0.5 * mu_max,
        0.25 * mu_min + 0.75 * mu_max,
        mu_max,
    ]

    for mu in mu_list:
        print(f"mu: {mu:3.2f}")
        # measuring the min and max coordinates of the bounding box
        theta = torch.linspace(0, 2 * np.pi, 10_000, dtype=torch.float64)[:, None]
        x = rho_max * torch.cos(theta)
        y = rho_max * torch.sin(theta)
        mu_ = mu * torch.ones_like(x)
        xT_pred, yT_pred = apply_symplecto(x, y, mu_)
        xT_ex, yT_ex = apply_exact_symplecto(x, y, mu_)
        xT_min, xT_max = xT_pred.min().item(), xT_pred.max().item()
        yT_min, yT_max = yT_pred.min().item(), yT_pred.max().item()
        lx = xT_max - xT_min
        ly = yT_max - yT_min

        # draw the contours
        _, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

        ax.scatter(xT_ex.detach().cpu(), yT_ex.detach().cpu(), s=25, c="red")
        ax.scatter(xT_pred.detach().cpu(), yT_pred.detach().cpu(), s=1, c="green")
        ax.set_aspect("equal")
        if save_plots:
            plt.savefig(name + f"_mu{mu:3.2f}.pdf")

        plt.show()

        print(f"Haussdorf distance: {get_hausdorff_distance(mu):3.2e}")


def param_shape_superposition(
    rho_max, mu_min, mu_max, apply_symplecto, save_plots, name
):
    import numpy as np
    import torch

    mu_list = [
        mu_min,
        0.75 * mu_min + 0.25 * mu_max,
        0.5 * mu_min + 0.5 * mu_max,
        0.25 * mu_min + 0.75 * mu_max,
        mu_max,
    ]

    lx, ly = 0, 0

    for mu in mu_list:
        # measuring the min and max coordinates of the bounding box
        theta = torch.linspace(0, 2 * np.pi, 10_000, dtype=torch.float64)[:, None]
        x = rho_max * torch.cos(theta)
        y = rho_max * torch.sin(theta)
        mu_ = mu * torch.ones_like(x)
        xT_pred, yT_pred = apply_symplecto(x, y, mu_)
        xT_min, xT_max = xT_pred.min().item(), xT_pred.max().item()
        yT_min, yT_max = yT_pred.min().item(), yT_pred.max().item()
        lx = max(lx, xT_max - xT_min)
        ly = max(ly, yT_max - yT_min)

    # draw the contours
    _, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    for mu in mu_list:
        # measuring the min and max coordinates of the bounding box
        theta = torch.linspace(0, 2 * np.pi, 10_000, dtype=torch.float64)[:, None]
        x = rho_max * torch.cos(theta)
        y = rho_max * torch.sin(theta)
        mu_ = mu * torch.ones_like(x)
        xT_pred, yT_pred = apply_symplecto(x, y, mu_)

        ax.scatter(
            xT_pred.detach().cpu(),
            yT_pred.detach().cpu(),
            s=1,
        )

    if save_plots:
        plt.savefig(name + "_superposition.pdf")
    plt.show()


def optimality_condition(get_optimality_condition, save_plots, name):
    n_pts = 10_000
    if device == "cpu":
        n_pts = 1_000

    optimality_condition, xT, yT = get_optimality_condition(n_pts)
    xT_min, xT_max = xT.min().item(), xT.max().item()
    yT_min, yT_max = yT.min().item(), yT.max().item()
    lx = xT_max - xT_min
    ly = yT_max - yT_min

    optimality_condition = optimality_condition - optimality_condition.mean()

    # draw the contours
    fig, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    im = ax.scatter(
        xT.detach().cpu(),
        yT.detach().cpu(),
        s=1,
        c=optimality_condition.detach().cpu(),
        cmap="turbo",
    )

    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    if save_plots:
        plt.savefig(name + ".pdf")

    plt.show()


def optimality_condition_param(
    mu_min, mu_max, get_optimality_condition, save_plots, name
):
    mu_list = [
        mu_min,
        0.75 * mu_min + 0.25 * mu_max,
        0.5 * mu_min + 0.5 * mu_max,
        0.25 * mu_min + 0.75 * mu_max,
        mu_max,
    ]

    lx, ly = 0, 0

    for mu in mu_list:
        optimality_condition, xT, yT = get_optimality_condition(mu)
        xT_min, xT_max = xT.min().item(), xT.max().item()
        yT_min, yT_max = yT.min().item(), yT.max().item()
        lx = max(lx, xT_max - xT_min)
        ly = max(ly, yT_max - yT_min)

    # draw the contours
    fig, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    for mu in mu_list:
        optimality_condition, xT, yT = get_optimality_condition(mu)
        xT_min, xT_max = xT.min().item(), xT.max().item()
        yT_min, yT_max = yT.min().item(), yT.max().item()
        optimality_condition = optimality_condition - optimality_condition.mean()
        im = ax.scatter(
            xT.detach().cpu(),
            yT.detach().cpu(),
            s=1,
            c=optimality_condition.detach().cpu(),
            cmap="turbo",
        )

    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    if save_plots:
        plt.savefig(name + "_superposition.pdf")
    plt.show()


def deep_shape_error(
    rho_max,
    apply_symplecto,
    xT_ex,
    yT_ex,
    save_plots,
    name,
):
    import numpy as np
    import torch

    n_pts = 1000

    # measuring the min and max coordinates of the bounding box
    theta = torch.linspace(0, 2 * np.pi, n_pts, dtype=torch.float64)[:, None]
    x = rho_max * torch.cos(theta)
    y = rho_max * torch.sin(theta)
    xT_pred, yT_pred = apply_symplecto(x, y)
    xT_pred, yT_pred = translate_to_zero(xT_pred, yT_pred, n_pts)
    xT_min, xT_max = xT_pred.min().item(), xT_pred.max().item()
    yT_min, yT_max = yT_pred.min().item(), yT_pred.max().item()
    lx = xT_max - xT_min
    ly = yT_max - yT_min

    # draw the contours
    _, ax = plt.subplots(1, 1, figsize=(10 * lx / ly, 10 * ly / lx))

    ax.scatter(
        xT_ex,
        yT_ex,
        s=25,
        c="red",
    )
    ax.scatter(
        xT_pred.detach().cpu(),
        yT_pred.detach().cpu(),
        s=1,
        c="green",
    )

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
