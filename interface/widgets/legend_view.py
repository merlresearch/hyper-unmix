# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QButtonGroup, QCheckBox, QGraphicsProxyWidget
from pyqtgraph.graphicsItems.LegendItem import ItemSample


class LegendItem(pg.LegendItem):

    clicked = QtCore.pyqtSignal(int)

    def __init__(self, *args, ui_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._group = QButtonGroup()
        self.ui_callback = ui_callback

    def addItem(self, item, name):
        widget = QCheckBox(name)
        palette = widget.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColorConstants.Transparent)
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColorConstants.Black)
        widget.setPalette(palette)
        row = self.layout.rowCount()
        widget.clicked.connect(lambda: self.button_clicked(row))
        proxy = item.scene().addWidget(widget)
        if isinstance(item, ItemSample):
            sample = item
        else:
            sample = ItemSample(item)
        self.layout.addItem(proxy, row, 0)
        self.layout.addItem(sample, row, 1)
        self.items.append((proxy, sample))
        self.updateSize()

    def uncheck_all(self):
        for item in self.allChildItems():
            if isinstance(item, QGraphicsProxyWidget):
                item.widget().setCheckState(QtCore.Qt.CheckState.Unchecked)

    def button_clicked(self, item_num):
        self.ui_callback(item_num)
