# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pyqtgraph as pg
from matplotlib import cm


class SpecPanel(pg.GraphicsLayoutWidget):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.setWindowTitle("Spec. Panel")
        self.setBackground("w")

        vb = self.addViewBox()
        self.audio = pg.ImageItem(axisOrder="row-major")
        self.selection = pg.ImageItem(axisOrder="row-major")
        vb.addItem(self.audio)
        vb.addItem(self.selection)

        # Get the colormap
        cm_1 = cm.get_cmap("binary")
        cm_2 = cm.get_cmap("winter")
        cm_1._init()
        cm_2._init()
        lut1, self.lut_sel = (cm_1._lut * 255).view(np.ndarray), (cm_2._lut * 255).view(np.ndarray)
        self.audio.setLookupTable(lut1)

    def update_audio_mesh(self, data):
        self.audio.setImage(data.T)

    def update_selection_mesh(self, data):
        data = data.T  # txf
        d = self.lut_sel[data.flatten(), :].reshape(*data.shape, 4)
        d[data == 0, -1], d[data != 0, -1] = 0, 255
        self.selection.setImage(d.astype(int))
