# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from argparse import Namespace
from pathlib import Path

import torch
import yaml

from hyperbolic_separator import HyperbolicSigSep

torch.manual_seed(0)
DEFAULT_CONF_PATH = Path("hyperbolic") / "model" / "conf.yaml"


def _build_separator():
    with open(DEFAULT_CONF_PATH, "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        hparams = Namespace(**params)
    return HyperbolicSigSep(hparams=hparams)


def test_separated_signals_have_same_output_length():
    model = _build_separator()
    n_batch, n_samples = 5, 20000
    signal = torch.rand((n_batch, n_samples))
    wf_estimates = model.separate(signal)
    assert wf_estimates.shape[-1] == n_samples
