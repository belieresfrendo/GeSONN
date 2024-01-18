def make_colormap():
    import numpy as np

    colors = np.empty((8, 3))

    colors[0] = np.array((0.278431372549, 0.278431372549, 0.858823529412))
    colors[1] = np.array((0.000000000000, 0.000000000000, 0.360784313725))
    colors[2] = np.array((0.000000000000, 1.000000000000, 1.000000000000))
    colors[3] = np.array((0.000000000000, 0.501960784314, 0.000000000000))
    colors[4] = np.array((1.000000000000, 1.000000000000, 0.000000000000))
    colors[5] = np.array((1.000000000000, 0.380392156863, 0.000000000000))
    colors[6] = np.array((0.419607843137, 0.000000000000, 0.000000000000))
    colors[7] = np.array((0.878431372549, 0.301960784314, 0.301960784314))

    opacities = np.arange(8) / 7

    list_for_colormap = [(opacities[i], colors[i]) for i in range(8)]

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("rainbow desaturated", list_for_colormap)