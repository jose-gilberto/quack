"""Color and marker utilities shared across the `quack.visualization` module.

Provides a colorblind-safe base palette (Okabe & Ito, 2008) plus a
deterministic strategy to extend it to an arbitrary number of methods,
so multi-method comparison plots remain readable regardless of how many
quantifiers are being compared.
"""
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito palette: perceptually distinct and safe for the most common
# forms of color vision deficiency (protanopia, deuteranopia, tritanopia).
# Ref: Okabe, M. & Ito, K. (2008). "Color Universal Design (CUD)".
COLORBLIND_PALETTE: list[str] = [
  '#0072B2',  # blue
  '#D55E00',  # vermillion
  '#009E73',  # bluish green
  '#CC79A7',  # reddish purple
  '#E69F00',  # orange
  '#56B4E9',  # sky blue
  '#F0E442',  # yellow
  '#000000',  # black
]

# Reserved for elements that should always stand out regardless of the
# number of methods being plotted (reference/diagonal lines, box edges).
REFERENCE_COLOR: str = '#404040'

MARKERS: list[str] = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "8"]

def get_color_palette(n_colors: int, palette: Sequence = None) -> list:
  """Build a list of `n_colors` visually distinct colors.

  Falls back to the colorblind-safe base palette while there are enough
  colors available. When more colors than the base palette are requested
  (e.g. many quantifiers being compared at once), it extends the palette
  by uniformly sampling a perceptually-uniform colormap so that all
  colors remain distinguishable from each other.

  Parameters
  ----------
  n_colors: int
    Number of distinct colors needed.
  palette: Sequence, default = None
    User-provided palette (hex strings or RGBA tuples) to use instead of
    the default colorblind-safe one.

  Returns
  -------
  colors_list: list
    List of length `n_colors` containing hex strings or RGBA tuples.
  """
  base = list(palette) if palette is not None else list(COLORBLIND_PALETTE)

  if n_colors <= len(base):
    return base[:n_colors]

  # extend deterministically using a perceptually-uniform colormap so
  # additional colors stay maximally separated from one another
  cmap = plt.get_cmap('turbo')
  n_extra = n_colors - len(base)
  extra = [cmap(x) for x in np.linspace(0.05, 0.95, n_extra)]
  return base + extra


def get_marker_cycle(n_markers: int, markers: Sequence[str] = None) -> list:
  """Cycle through a fixed list of distinguishable marker shapes.

  Parameters
  ----------
  n_markers: int
    Number of markers needed.
  markers: Sequence[str], default = None
    Custom marker list.

  Returns
  -------
  markers_shapes: list
    List of length `n_markers` with matplotlib marker style strings.
  """
  base = list(markers) if markers is not None else list(MARKERS)
  return [base[i % len(base)] for i in range(n_markers)]
