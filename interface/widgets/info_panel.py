# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pyqtgraph as pg


class InfoPanel(pg.GraphicsLayoutWidget):
    def __init__(self, logger, freq_res, dist_res=100):
        super().__init__()
        self.logger = logger
        self.setWindowTitle("Info Panel")
        self.setBackground("w")

        self.freq_res = freq_res
        self.dist_res = dist_res

        # Add sphere
        self.pFreq = self.addPlot(title="Selected Bins by Frequency", row=0, col=0)
        self.plotFreqDistrib([])

        # Add sphere
        self.pCert = self.addPlot(title="Selected Bins by Certainty", row=1, col=0)
        self.plotCertaintyDistrib([])

    def updateHistograms(self, data):
        self.plotCertaintyDistrib(data["dist"])
        self.plotFreqDistrib(data["freq"])

    def plotCertaintyDistrib(self, data):
        self.pCert.clear()
        y, x = np.histogram(data, bins=self.dist_res)
        self.pCert.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))

    def plotFreqDistrib(self, data):
        self.pFreq.clear()
        self.pFreq.plot(
            np.arange(0, len(data) + 1), data, stepMode="center", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150)
        )
