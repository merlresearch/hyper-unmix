# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from PyQt6 import QtCore
from PyQt6.QtWidgets import QPlainTextEdit


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._process = QtCore.QProcess()
        self._process.readyReadStandardOutput.connect(self.handle_stdout)
        self._process.readyReadStandardError.connect(self.handle_stderr)

        self.setDocumentTitle("Logger")
        self.add_log("All console logs display here ...")

    def start_log(self, program, arguments=None):
        if arguments is None:
            arguments = []
        self._process.start(program, arguments)

    def add_log(self, message):
        self.appendPlainText(message.rstrip())

    def handle_stdout(self):
        message = self._process.readAllStandardOutput().data().decode()
        self.add_log(message)

    def handle_stderr(self):
        message = self._process.readAllStandardError().data().decode()
        self.add_log(message)
