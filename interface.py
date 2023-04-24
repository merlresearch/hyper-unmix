# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path

import pyqtgraph as pg
import pyqtgraph.parametertree as ptree
import torch
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QSplitter
from pyqtgraph.parametertree import parameterTypes as ptypes

from hyperbolic.hypertools.hypernn import certainty_from_coors
from hyperbolic.model.model import MaskInference
from interface.widgets import audio_player, audio_processor, ball, info_panel, logview, spec_panel
from lsx_dataset import SOURCE_NAMES_CHILDREN, SOURCE_NAMES_PARENT
from separate import DEFAULT_PRE_TRAINED_MODEL_PATH

srcs_groups = deepcopy(SOURCE_NAMES_CHILDREN)
srcs_groups.insert(0, SOURCE_NAMES_PARENT)
SOURCE_NAMES = [src for group in srcs_groups for src in group]

pg.setConfigOptions(imageAxisOrder="row-major")
logger = None

"""
###############################################################################
#########################      Main Window     ################################
###############################################################################
"""


class Window(QMainWindow):
    def __init__(self, proc, logger, **kwargs):
        super().__init__()

        self.logger = logger
        self.proc = proc
        self.kwargs = kwargs
        self.setWindowTitle("Selective Hyperbolic Source Separation")
        self.setGeometry(0, 0, 3000, 3000)
        self.UiComponents()
        self.show()

    def UiComponents(self):

        translate = QCoreApplication.translate
        splitter = QSplitter()

        # Left-most window
        params = ptree.Parameter.create(
            name=translate("ScatterPlot", "Parameters"),
            type="group",
            children=[
                dict(name="load", title=translate("ScatterPlot", "Load Audio File"), type="action"),
                dict(
                    name="db_threshold",
                    title=translate("ScatterPlot", "Scatter Display Threshold (dB)"),
                    type="slider",
                    limits=[-90, 20],
                    value=-5,
                    step=1,
                ),
                dict(
                    name="cert_threshold",
                    title=translate("ScatterPlot", "Certainty Synthesis Threshold"),
                    type="slider",
                    limits=[0, 1.0],
                    value=0.0,
                    step=0.01,
                ),
                dict(name="project", title=translate("ScatterPlot", "Project Audio"), type="action"),
                dict(name="geodesics", title=translate("ScatterPlot", "Class Geodesics"), type="bool", value=False),
                dict(
                    name="intersections",
                    title=translate("ScatterPlot", "Geodesics Intersections"),
                    type="bool",
                    value=False,
                ),
            ],
        )
        for c in params.children():
            c.setDefault(c.value())
            if type(c) in [ptypes.ActionParameter]:
                c.sigActivated.connect(self.action_event)
            elif type(c) in [ptypes.SimpleParameter]:
                c.sigValueChanged.connect(self.action_event)
            elif type(c) == ptypes.SliderParameter:
                c.sigValueChanged.connect(self.sig_changed)

        pt = ptree.ParameterTree(showHeader=False)
        pt.setParameters(params)

        self.spec = spec_panel.SpecPanel(logger=logger)
        rightSplit = QSplitter(Qt.Orientation.Vertical)
        rightSplit.addWidget(pt)
        rightSplit.addWidget(self.spec)
        rightSplit.setSizes([500, 500])

        splitter.addWidget(rightSplit)

        # Middle window
        mid_window_size = (490, 490)
        self.ball = ball.BallView(
            logger=self.logger,
            selection_callback=self.selection_callback,
            geodesics=self.proc.get_model_geodesics(),
            ball=self.proc.model.mask_layer.mlr.ball,
            source_names=SOURCE_NAMES,
            parent=self,
            show=True,
            size=mid_window_size,
            border=True,
            title="Source Sep. on the Hyperbolic Disk",
        )
        # Create ap panel
        self.mix_ap = audio_player.AudioPlayer(
            logger=self.logger, width=mid_window_size[0] // 2, height=10, text_label="Loaded audio File: "
        )
        self.sel_ap = audio_player.AudioPlayer(
            logger=self.logger, width=mid_window_size[0] // 2, height=10, text_label="Poincare Selection: "
        )
        self.ap_splitter = QSplitter()
        self.ap_splitter.addWidget(self.mix_ap)
        self.ap_splitter.addWidget(self.sel_ap)

        midSplit = QSplitter(Qt.Orientation.Vertical)
        midSplit.addWidget(self.ball)
        midSplit.addWidget(self.ap_splitter)
        midSplit.setSizes([800, 100])
        splitter.addWidget(midSplit)

        # Right-most window
        rightSplit = QSplitter(Qt.Orientation.Vertical)
        self.info_pan = info_panel.InfoPanel(logger=self.logger, freq_res=self.proc.stft_kwargs["fft_size"] // 2 + 1)
        rightSplit.addWidget(self.info_pan)
        rightSplit.addWidget(self.logger)
        rightSplit.setSizes([500, 500])
        splitter.addWidget(rightSplit)

        # setting this layout to the widget
        splitter.setSizes([300, 800, 300])
        self.setCentralWidget(splitter)

    # Buttons
    def action_event(self, param):
        if param.name() == "load":
            mix_path, audio_set = self.fileBrowsing()
            if audio_set:
                self.set_player_source(mix_path, idx=0)
                self.display_audio()
        elif param.name() == "project":
            self.proc.project_audio()
            self.ball.set_scatter_points(self.proc.get_data_points_as_xy())
        elif param.name() == "geodesics":
            self.ball.toggle_geodesics(param.value())
        elif param.name() == "intersections":
            self.ball.set_geo_intersections_bool(param.value())

    # Slider
    def sig_changed(self, param):
        if param.name() == "db_threshold":
            self.proc.set_db_threshold(param.value())
        elif param.name() == "cert_threshold":
            self.proc.set_cert_threshold(param.value())

    # Browse for file to process
    def fileBrowsing(self):
        filePath = QFileDialog.getOpenFileName(self, "", "Desktop", "*.wav")
        audio_set = False
        if filePath != "" and filePath[0] != "":
            self.proc.set_audio(filePath[0])
            self.logger.add_log("Loaded file with path: {}".format(filePath[0]))
            audio_set = True

        return filePath[0], audio_set

    def set_player_source(self, path, idx=0):
        self.ap_splitter.widget(idx).set_audio_file_path(path)

    def display_audio(self):
        self.spec.update_audio_mesh(self.proc.get_mesh_spec())

    def render_selection(self):
        idxs, _ = self.ball.get_current_selection()
        render_path = "./interface/audio/tmp.wav"
        self.set_player_source(None, idx=1)
        _ = self.proc.synthesize_selection(selection=idxs, to_disk=True, path=render_path)
        self.set_player_source(render_path, idx=1)
        self.display_audio()

    # get notification from
    def selection_callback(self, idxs, coors):
        # update info panel
        freq = self.proc.get_mask_from_selection(idxs).squeeze(-1).detach().numpy()
        freq_count = freq.sum(0)
        dists = certainty_from_coors(coors)
        self.info_pan.updateHistograms({"freq": freq_count, "dist": dists})

        # update specotrgram panel
        self.spec.update_selection_mesh((freq * 255).astype(int))

        self.render_selection()


"""
###############################################################################
#########################     Main Function    ################################
###############################################################################
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_PRE_TRAINED_MODEL_PATH,
        type=Path,
        help="The checkpoint model to load with the interface.",
    )
    args = parser.parse_args()

    pg.setConfigOptions(antialias=True)

    # Create GUI and instantiate logger
    app = pg.mkQApp("Hyperbolic Source Separation")
    logger = logview.LogView()
    logger.resize(100, 50)
    logger.start_log("adb")

    state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))["state_dict"]
    weights = {k.replace("model.", ""): v for k, v in state_dict.items()}
    params = torch.load(args.checkpoint, map_location=torch.device("cpu"))["hyper_parameters"]
    hparams = Namespace(**params)

    # Load model
    model = MaskInference(**hparams.model)
    model.load_state_dict(weights)
    model.eval()

    # Create tmp directory
    Path("./audio").mkdir(parents=True, exist_ok=True)

    # Create processor
    proc = audio_processor.Processor(logger=logger, model=model, stft_kwargs=hparams.features)

    window = Window(proc=proc, logger=logger, **vars(hparams))

    # Start the program
    pg.exec()
