import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["fontenc"], "size": 15})
rc("text", usetex=True)


def loss(loss_history):
    _, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_yscale("symlog", linthresh=1e-4)
    plt.show()


def edp(x, y, u, title):
    fig, ax = plt.subplots()
    im = ax.scatter(
        x,
        y,
        s=1,
        c=u,
        cmap="gist_ncar",
        # cmap=colormaps.make_colormap(),
    )
    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    ax.set_title(title)
    # ax.legend()

    plt.show()

def shape(x, y, title=None):
    _, ax = plt.subplots()
    ax.scatter(
        x,
        y,
        s=1,
    )
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)

    plt.show()

def shape_error(x, y, u, v, title=None):
    _, ax = plt.subplots()
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

    plt.show()

def edp_shape_error(edp, x, y, u, v, title=None):
    fig, ax = plt.subplots()
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

    plt.show()