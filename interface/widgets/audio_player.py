# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class AudioPlayer(QWidget):
    def __init__(self, logger, width, height, path=None, text_label=None):
        super().__init__()
        self.logger = logger
        self.path = path
        self.text_label = text_label
        self.window_width, self.window_height = width, height
        self.setMinimumSize(self.window_width, self.window_height)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel(text=text_label)
        self.layout.addWidget(self.label)

        btn_play = QPushButton("Play", clicked=self.playAudioFile)
        self.layout.addWidget(btn_play)

        btn_stop = QPushButton("Stop", clicked=self.stopAudioFile)
        self.layout.addWidget(btn_stop)

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

    def volumeUp(self):
        currentVolume = self.player.volume()  #
        self.player.setVolume(currentVolume + 5)

    def volumeDown(self):
        currentVolume = self.player.volume()  #
        self.player.setVolume(currentVolume - 5)

    def volumeMute(self):
        self.player.setMuted(not self.player.isMuted())

    def set_audio_file_path(self, path):
        self.path = path
        if path is not None:
            self.path = path if os.path.isabs(path) else os.path.abspath(path)
            self.label.setText(self.text_label + path)
        url = QUrl.fromLocalFile(self.path)
        self.player.setSource(url)

    def playAudioFile(self):
        if self.player.source() == QUrl(None):
            self.logger.add_log(
                "Path/Selection has not been set yet. "
                "Please Load an audio file first or make your selection in Poincaré space."
            )
        self.player.play()

    def stopAudioFile(self):
        self.player.stop()
