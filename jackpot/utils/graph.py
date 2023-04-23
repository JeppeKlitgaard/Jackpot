import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def show(arr, title):
    """
    Show a 2D array as a plot.
    """
    plt.figure()
    im = plt.imshow(arr, interpolation=None)
    plt.title(title)

    # Legends
    values = np.unique(arr.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [
        mpatches.Patch(color=colors[i], label=f"{values[i]}")
        for i in range(len(values))
    ]
    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        framealpha=0.2,
    )
