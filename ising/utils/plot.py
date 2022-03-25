import numpy as np
import matplotlib

def make_alpha(cmap):
  my_cmap = cmap(np.arange(cmap.N))
  my_cmap[:,-1] = np.linspace(0, 1, cmap.N)**3
  return matplotlib.colors.ListedColormap(my_cmap)