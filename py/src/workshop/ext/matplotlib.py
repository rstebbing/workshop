##########################################
# File: matplotlib.py                    #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import numpy as np
from matplotlib import patches as mpatches
from mpl_toolkits.mplot3d.proj3d import proj_transform


class FancyArrow3DPatch(mpatches.FancyArrowPatch):
    """https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c"""

    def __init__(self, xyz, dxdydz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)

        self._xyz = xyz
        self._dxdydz = dxdydz

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, _ = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
