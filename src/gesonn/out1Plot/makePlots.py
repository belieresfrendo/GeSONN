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


def edp(x, y, u, save_plots, name, title=None, figsize=(7.5,7.5)):
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

def shape(x, y, save_plots, name, title=None):
    _, ax = plt.subplots(figsize=(7.5,7.5))
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
    _, ax = plt.subplots(figsize=(5,5))
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
    _, ax = plt.subplots(figsize=(7.5,7.5))
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
    fig, ax = plt.subplots(figsize=(7.5,7.5))
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
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', borderaxespad=0., ncol=2)
    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if save_plots:
        plt.savefig(name + "_edp_error.pdf")

    plt.show()