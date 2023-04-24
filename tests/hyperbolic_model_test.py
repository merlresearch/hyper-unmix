# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from hyperbolic.model.model import MaskInference

torch.manual_seed(0)


def test_model_output_shape():
    n_sources, n_layers, hidden_dim, embed_dim, n_freq_bins = 2, 3, 30, 2, 257
    model = MaskInference(n_sources, n_layers, hidden_dim, n_freq_bins, hyperbolic_dim=embed_dim)
    nbatch, nframes = 5, 200
    input = torch.rand((nbatch, nframes, n_freq_bins))
    mlr_output, embeddings = model(input)
    expected_shape_mlr = (nbatch, nframes, n_freq_bins, n_sources)
    expected_shape_embeddings = (nbatch * nframes * n_freq_bins, embed_dim)
    assert mlr_output.shape == expected_shape_mlr
    assert embeddings.shape == expected_shape_embeddings
