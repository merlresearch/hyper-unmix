# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from hyperbolic.audio_utils import find_invalid_sources, istft, si_snr, stft

torch.manual_seed(0)


def test_si_snr_scale_invariant():
    n_batch, n_srcs, n_samples = 5, 3, 2000
    targets = torch.rand((n_batch, n_srcs, n_samples))
    estimation_errors = 0.1 * torch.rand(targets.shape)
    estimates = targets + estimation_errors
    estimates_scale = 0.5 * estimates
    loss_no_scale = si_snr(estimates, targets)
    loss_scale = si_snr(estimates_scale, targets)
    torch.testing.assert_close(loss_scale, loss_no_scale)


def test_stft_perfect_reconstruction_sources():
    n_batch, n_srcs, n_samples = 5, 3, 20000
    signal = torch.rand((n_batch, n_srcs, n_samples))
    sig_stft = stft(signal)
    sig_stft = sig_stft.permute(0, 2, 3, 1)  # istft expects source dimension in the back
    signal_hat = istft(sig_stft, signal_length=n_samples)
    torch.testing.assert_close(signal_hat, signal, rtol=1.0e-5, atol=1.0e-6)


def test_stft_perfect_reconstruction_mixture():
    n_batch, n_samples = 5, 20000
    signal = torch.rand((n_batch, n_samples))
    sig_stft = stft(signal)
    signal_hat = istft(sig_stft, signal_length=n_samples)
    torch.testing.assert_close(signal_hat, signal, rtol=1.0e-5, atol=1.0e-6)


def test_find_invalid_sources():
    n_batch, n_srcs, n_samples = 5, 3, 20000
    src_signals = torch.rand((n_batch, n_srcs, n_samples))
    src_signals[2, 0:2, :] = 0.0  # all sources except one are silent, which equals the mixture, so all are invalid
    src_signals[4, 1, :] = 0.0  # a single silent source
    mix_signals = src_signals.sum(dim=1, keepdims=True).repeat(1, n_srcs, 1)
    nan_mat_est = find_invalid_sources(mix_signals, src_signals)
    nan_mat_true = torch.ones((n_batch, n_srcs))
    nan_mat_true[2, :] = torch.nan
    nan_mat_true[4, 1] = torch.nan
    torch.testing.assert_close(nan_mat_est, nan_mat_true, equal_nan=True)
