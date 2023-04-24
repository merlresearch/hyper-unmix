# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Optional, Tuple

import torch
import torch.nn as nn

from hyperbolic.hypertools.hypernn import EuclMLR, MobiusMLR
from hyperbolic.hypertools.hyptorch import ToPoincare


class _RecurrentStack(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_input: int,
        n_output: int,
        dropout: Optional[float] = 0.0,
        bidirectional: Optional[bool] = False,
    ) -> None:
        """
        Creates a stack of LSTMs used to process an audio sequence represented as (sequence_length, num_features).
        Args:
            n_layers: (int) Number of layers in stack.
            n_input: (int) Number of features being mapped for each frame. Usually equates to `n_freq`
            n_output: (int) Hidden size of recurrent stack for each layer.
            dropout: (float) Dropout between layers.
            bidirectional: (bool) True makes this a BiLSTM. Note that this doubles the hidden size.
        """

        super().__init__()

        self.norm_layer = nn.BatchNorm1d(n_input)
        self.LSTM = nn.LSTM(
            n_input, n_output, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a tensor of shape (n_batch, n_frames, n_input)
        :return: y a tensor of shape (n_batch, n_frames, n_output)
        """
        x = torch.transpose(x, 1, 2)
        x = self.norm_layer(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.LSTM(x)

        return x


class _MaskLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_sources: int,
        n_freq_bins: int,
        hyperbolic_dim: Optional[int] = 2,
        hyperbolic_train: Optional[bool] = False,
        hyperbolic_k: Optional[int] = 1.0,
    ):
        """
        Predicts source-wise real-valued TxF masks from input embeddings. In this case the input features are first
        projected onto an hyperbolic manifold (i.e., Poincaré Ball). The masks are then obtained by mean of
        Multinomial Logistic Regression (MLR).
        Args:
            input_dim: (int) Input feature dimension (e.g., hidden feature dim. from LSTM network)
            n_sources: (int) Number of expected masks to predict in MLR
            n_freq_bins: (int) The number of expected frequency bins for the masks (i.e., STFT//2+1)
            hyperbolic_dim: (int) The number of dimension of the Poincaré Ball
            hyperbolic_train: (bool) Whether or not to train the manifold curvature coefficient
            hyperbolic_k: (float) The hyperbolic curvature coefficient. Must be positive for the Poincaré Ball.
        """

        super().__init__()

        self.n_sources = n_sources
        self.n_freq_bins = n_freq_bins
        self.hyperbolic_dim = hyperbolic_dim
        self.hyperbolic_k = hyperbolic_k

        self.linear = nn.Linear(input_dim, n_freq_bins * self.hyperbolic_dim)

        if hyperbolic_k > 0.0:
            self.tp = ToPoincare(
                c=hyperbolic_k, train_x=False, train_c=hyperbolic_train, ball_dim=self.hyperbolic_dim, riemannian=False
            )
            self.mlr = MobiusMLR(self.hyperbolic_dim, self.n_sources, c=hyperbolic_k)
        else:
            self.tp = lambda x: x
            self.mlr = EuclMLR(self.hyperbolic_dim, self.n_sources)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a tensor of shape (n_batch, n_frames, n_hid)
        :return: y a tensor of shape (n_batch, n_frames, n_freqs, n_sources)
        """
        n_batch, n_frames, input_dim = x.shape

        x = self.linear(x)
        x = x.double()
        h = self.tp(x.reshape(-1, self.hyperbolic_dim))
        x = self.mlr(h)
        x = x.reshape(n_batch, n_frames, self.n_freq_bins, self.n_sources)
        return x, h


class MaskInference(nn.Module):
    def __init__(
        self,
        n_sources: int,
        n_layers: int,
        hidden_dim: int,
        n_freq_bins: int,
        dropout: Optional[float] = 0.0,
        bidirectional: Optional[bool] = False,
        hyperbolic_dim: Optional[int] = 2,
        hyperbolic_train: Optional[bool] = False,
        hyperbolic_k: Optional[float] = 1.0,
        **kwargs
    ):
        """
        Predicts source-wise real-valued TxF masks from an input mixture in the T-F domain.
        In this case the input features are first embedded into an LSTM stack. These embeddings
        are then projected onto an hyperbolic manifold (i.e., Poincaré Ball). The masks are then obtained by mean of
        Multinomial Logistic Regression (MLR).
        Args:
            n_sources: (int) Number of expected masks to predict in MLR
            n_layers: (int) Number of layers in stack.
            hidden_dim: (int) Stack input feature dimension (e.g., hidden feature dim. from LSTM network)
            n_freq_bins: (int) The number of expected frequency bins for the masks (i.e., STFT//2+1)
            dropout: (Optional, float) Dropout between layers.
            bidirectional: (Optional, bool) True makes this a BiLSTM. Note that this doubles the hidden size.
            hyperbolic_dim: (Optional, int) The number of dimension of the Poincaré Ball
            hyperbolic_train: (Optional, bool) Whether or not to train the manifold curvature coefficient
            hyperbolic_k: (Optional, float) The hyperbolic curvature coefficient.
            Must be positive for the Poincaré Ball.
        """

        super().__init__()

        self.lstm = _RecurrentStack(n_layers, n_freq_bins, hidden_dim, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_dim *= 2
        self.mask_layer = _MaskLayer(
            hidden_dim,
            n_sources,
            n_freq_bins,
            hyperbolic_dim=hyperbolic_dim,
            hyperbolic_train=hyperbolic_train,
            hyperbolic_k=hyperbolic_k,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        :param x: a tensor of shape (n_batch, n_frames, n_freqs)
        :return: a tuple of two torch tensors
                 y tensor of shape (n_batch, n_frames, n_freqs, n_sources) containing the MLR output before softmax
                 h tensor of shape (n_batch*n_frames*n_freqs, hyperbolic_dim) containing the embedding for each TF bin
        """
        x = self.lstm(x)
        y, h = self.mask_layer(x)

        return y, h
