# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pyqtgraph as pg
import torch

from hyperbolic.hypertools.dist2plane import distance2plane
from interface.utils import add_geodesic_grid, add_geodesics


class BallView(pg.GraphicsLayoutWidget):
    def __init__(self, logger, selection_callback, geodesics, ball, source_names, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logger
        self.setWindowTitle("Selective Hyperbolic Source Sep.")
        self.setBackground("w")

        # Communicate with main window
        self.selection_callback = selection_callback

        # Add sphere
        self.p = self.addPlot(title="")
        p_ellipse = pg.QtWidgets.QGraphicsEllipseItem(-1, -1, 2, 2)  # x, y, width, height
        p_ellipse.setPen(pg.mkPen(pg.mkColor("black"), width=4))
        self.p.setXRange(-1.25, 1.25)
        self.p.setYRange(-1.25, 1.25)
        self.p.addItem(p_ellipse)

        # Add geodesics
        self.geodesics = geodesics
        self.selected_geodesics = []
        self.geodesics_intersections = False
        self.source_names = source_names
        self.ball = ball
        self.class_geo = []
        self.legend = None
        add_geodesic_grid(self.p, ball, 0.5)

        self.selected_idxs = []
        self.last_selected = []

        # Make scatter selectable
        self.selectedPen = pg.mkPen("b", width=2)
        self.scatter = None
        self.create_new_scatter()

        self.rois = []
        self.rois.append(
            pg.EllipseROI(
                [-0.05, -0.05],
                [0.1, 0.1],
                parent=p_ellipse,
                hoverPen=pg.mkPen(pg.mkColor("green"), width=2),
                pen=pg.mkPen(pg.mkColor("black"), width=2),
                handlePen=pg.mkPen(pg.mkColor("blue"), width=3),
                handleHoverPen=pg.mkPen(pg.mkColor("m"), width=3),
            )
        )

        # Resize selector
        for roi in self.rois:
            roi.sigRegionChangeStarted.connect(self.started)
            roi.sigRegionChangeFinished.connect(self.finished)
            self.p.addItem(roi)

        self.update(self.rois[-1])

    def selected(self, points):
        for lp in self.last_selected:
            if lp is not None:
                lp.resetPen()
        for p in points:
            if p is not None:
                p.setPen(self.selectedPen)

        self.last_selected = points

    def toggle_geodesics(self, toggle):
        if toggle:
            if len(self.class_geo) > 0:
                [self.p.addItem(x) for x in self.class_geo]
            else:
                self.class_geo, self.legend = add_geodesics(
                    self.p,
                    self.ball,
                    p_k=self.geodesics[0],
                    a_k=self.geodesics[1],
                    line_width=2.0,
                    labels=self.source_names,
                    ui_callback=self.selection_from_geodesics,
                )
        else:
            [self.p.removeItem(x) for x in self.class_geo]

    def set_geo_intersections_bool(self, value):
        self.geodesics_intersections = value

    def selection_from_geodesics(self, item):
        # remove or add
        if item in self.selected_geodesics:
            self.selected_geodesics.remove(item)
        else:
            self.selected_geodesics.append(item)

        pts_all = np.array([[pt.pos().x(), pt.pos().y(), pt, i] for i, pt in enumerate(self.scatter.points())])
        coors = pts_all[:, :2].astype(np.float32)
        scatter = pts_all[:, 2:]
        dists = distance2plane(torch.from_numpy(coors), self.geodesics[0], self.geodesics[1], self.ball)

        # extract only pts within geodesics
        sels = np.where(dists[:, self.selected_geodesics] >= 0.0)[0]

        # if we want the interseciton
        if self.geodesics_intersections:
            uniques, counts = np.unique(sels, return_counts=True)
            sels = uniques[counts > 1]

        self.update(self.rois[-1], selected=scatter[sels, :])

    def set_scatter_points(self, pts):
        # Creat scatter plots
        self.create_new_scatter(pts)
        self.last_selected = []
        self.update(self.rois[-1])

    def create_new_scatter(self, pts=None):
        if self.scatter is not None:
            self.p.removeItem(self.scatter)
        self.scatter = pg.ScatterPlotItem(size=1, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.p.addItem(self.scatter)
        if pts is not None:
            xx, yy, cs, vis = pts["x"], pts["y"], pts["cs"], pts["visible"]
            spots = [{"pos": (x, y)} for x, y in zip(xx, yy)]
            self.scatter.setData(spots, brush=[pg.mkBrush(x) for x in cs])
            self.scatter.setPointsVisible(vis)

    # update colors and state of selected scatter points
    def update(self, roi, selected=None):
        if len(self.scatter.points()) > 0:
            # If selected is not passed, get list of all points inside roi
            if selected is None:
                # get ROI shape in coordinate system of the scatter plot
                roiShape = roi.mapToItem(self.scatter, roi.shape())
                selected = np.array(
                    [[pt, i] for i, pt in enumerate(self.scatter.points()) if roiShape.contains(pt.pos())]
                )

            selected_pts = np.array([])
            self.selected_idxs = np.array([])
            if len(selected) > 0:
                selected_pts = selected[:, 0]
                self.selected_idxs = selected[:, 1]

            # Highlight the points
            self.selected(selected_pts)
            self.selection_callback(*self.get_current_selection())

    def started(self, roi):
        if self.legend is not None:
            self.selected_geodesics = []
            self.legend.uncheck_all()
        self.selected(points=[])

    def finished(self, roi):
        self.update(roi)

    def get_current_selection(self):
        # convert spotitems to np coors
        coors = np.array([[c.pos().x(), c.pos().y()] for c in self.last_selected])
        return self.selected_idxs, coors
