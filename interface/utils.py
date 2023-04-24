# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import geoopt
import numpy as np
import pyqtgraph as pg
import seaborn as sns
import torch

from interface.widgets import legend_view


def generateRandomScatter(n=100):
    radius = 2 * np.pi
    r = (
        1
        * np.sqrt(
            np.random.rand(
                2,
                n,
            )
        )
        * 0.99
    )
    theta = np.random.rand(2, n) * radius * 0.99
    xx = 0 + r * np.cos(theta)
    yy = 0 + r * np.sin(theta)

    return xx, yy


def add_geodesic_grid(ax: pg.PlotItem, manifold: geoopt.Stereographic, line_width=0.1):

    # define geodesic grid parameters
    N_EVALS_PER_GEODESIC = 10000

    # get manifold properties
    K = manifold.k.item()
    R = manifold.radius.item()

    # get maximal numerical distance to origin on manifold
    if K < 0:
        # create point on R
        r = torch.tensor((R, 0.0), dtype=manifold.dtype)
        # project point on R into valid range (epsilon border)
        r = manifold.projx(r)
        # determine distance from origin
        max_dist_0 = manifold.dist0(r).item()
    else:
        max_dist_0 = np.pi * R
    # adjust line interval for spherical geometry
    circumference = 2 * np.pi * R

    # determine reasonable number of geodesics
    # choose the grid interval size always as if we'd be in spherical
    # geometry, such that the grid interpolates smoothly and evenly
    # divides the sphere circumference
    n_geodesics_per_circumference = 4 * 6  # multiple of 4!
    n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
    grid_interval_size = circumference / n_geodesics_per_circumference
    if K < 0:
        n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)

    # create time evaluation array for geodesics
    if K < 0:
        min_t = -1.2 * max_dist_0
    else:
        min_t = -circumference / 2.0
    t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]

    # define a function to plot the geodesics
    def plot_geodesic(gv):
        ax.plot(*gv.t().numpy())

    # define geodesic directions
    u_x = torch.tensor((0.0, 1.0))
    u_y = torch.tensor((1.0, 0.0))

    # add origin x/y-crosshair
    o = torch.tensor((0.0, 0.0))
    if K < 0:
        x_geodesic = manifold.geodesic_unit(t, o, u_x)
        y_geodesic = manifold.geodesic_unit(t, o, u_y)
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)

    # add geodesics per quadrant
    for i in range(1, n_geodesics_per_quadrant):
        i = torch.as_tensor(float(i))
        # determine start of geodesic on x/y-crosshair
        x = manifold.geodesic_unit(i * grid_interval_size, o, u_y)
        y = manifold.geodesic_unit(i * grid_interval_size, o, u_x)

        # compute point on geodesics
        x_geodesic = manifold.geodesic_unit(t, x, u_x)
        y_geodesic = manifold.geodesic_unit(t, y, u_y)

        # plot geodesics
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
        if K < 0:
            plot_geodesic(-x_geodesic)
            plot_geodesic(-y_geodesic)

    return ax


def add_geodesics(
    ax: pg.PlotItem,
    manifold: geoopt.Stereographic,
    p_k: torch.tensor,
    a_k: torch.tensor,
    line_width=0.1,
    labels=[],
    ui_callback=None,
):
    def detach(geodisic_list):
        return [x.detach().cpu() for x in geodisic_list]

    geo = []

    # Define some props for plot
    cs = (np.array(sns.color_palette("husl", n_colors=len(labels), as_cmap=False)) * 255).astype(int)
    legend = legend_view.LegendItem((100, 60), offset=(70, 20), labelTextSize="12pt", ui_callback=ui_callback)
    legend.setParentItem(ax)

    # Transform p_k/a_k
    a_k = a_k / a_k.norm(dim=-1, keepdim=True)
    points, direction = detach([p_k, a_k])
    brushes = [pg.mkBrush(x) for x in cs]

    orthogonal = torch.ones_like(a_k)
    orthogonal[:, 0] = -a_k[:, 1] / a_k[:, 0]

    arrows = []
    t = torch.linspace(-8, 8, 1000).unsqueeze(-1).repeat(1, 2).type_as(p_k)
    for i in range(points.shape[0]):
        geodesic = manifold.geodesic_unit(t, p_k[i, :], orthogonal[i, :]).detach().cpu()

        plot = ax.plot(x=geodesic[:, 0], y=geodesic[:, 1], pen=pg.mkPen(pg.mkColor(cs[i]), width=3), name=labels[i])

        # arrow geodesic directions as arrows
        arrow = pg.ArrowItem(
            angle=math.degrees(math.atan2(direction[i, 0].item(), direction[i, 1].item())),
            pen=pg.mkPen(pg.mkColor(cs[i]), width=3),
        )  # , direction[:, 0], direction[:, 1])
        arrow.setPos(points[i, 0].item(), points[i, 1].item())
        ax.addItem(arrow)

        # add plot+label to legend
        legend.addItem(plot, labels[i])

        # order matters, make sure the geodesics are the first added items
        geo.append(plot)
        arrows.append(arrow)

    # then add the directions
    geo.extend(arrows)

    # finally add the points of the geodesics
    plot1 = ax.plot(
        points[:, 0],
        points[:, 1],
        pen=None,
        symbolBrush=brushes,
        symbolPen=pg.mkPen(pg.mkColor("red"), width=4),
    )
    geo.append(plot1)

    return geo, legend
