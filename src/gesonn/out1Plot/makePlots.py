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
