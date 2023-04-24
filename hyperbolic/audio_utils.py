# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Optional

import torch

EPSILON = 1e-8


def si_snr(estimates: torch.Tensor, targets: torch.Tensor, dim: Optional[int] = -1) -> torch.Tensor:
    """
    Computes the negative scale-invariant signal (source) to noise (distortion) ratio.
    :param estimates (torch.Tensor): estimated source signals, tensor of shape [..., nsrcs, n_samples, ....]
    :param targets (torch.Tensor): ground truth signals, tensor of shape [...., nsrcs, n_samples, ....]
    :param mixture (torch.Tensor): mixture signals, tensor of shape [...., n_samples, ....]
    :param dim (int): time (sample) dimension
    :return (torch.Tensor): estimated SI-SNR with one value for each non-sample dimension
    """
    estimates = _mean_center(estimates, dim=dim)
    targets = _mean_center(targets, dim=dim)
    sig_power = l2_square(targets, dim=dim, keepdim=True)  # [n_batch, 1, n_srcs]
    dot_ = torch.sum(estimates * targets, dim=dim, keepdim=True)
    scale = dot_ / (sig_power + 1e-12)
    s_target = scale * targets
    e_noise = estimates - s_target
    si_snr_array = l2_square(s_target, dim=dim) / (l2_square(e_noise, dim=dim))
    si_snr_array = -10 * torch.log10(si_snr_array)
    return si_snr_array


def _mean_center(arr: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    mn = torch.mean(arr, dim=dim, keepdim=True)
    return arr - mn


def l2_square(arr: torch.Tensor, dim: Optional[int] = None, keepdim: Optional[bool] = False) -> torch.Tensor:
    return torch.sum(arr**2, dim=dim, keepdim=keepdim)


def find_invalid_sources(mix_signal: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Pre-processing step for SI-SDR that identifies silent sources, and sources that equal the mixture
    :param mix_signal (torch.Tensor): mixture signal replicated along the source dimension,
                                      tensor of shape [n_batch, n_srcs, n_samples]
    :param targets (torch.Tensor): ground truth source signals, tensor of shape [n_batch, n_srcs, n_samples]
    :return (torch.Tensor):  tensor of shape [n_batch, n_srcs] with 1 in valid source positions and nan otherwise
    """
    l2_sources = (l2_square(targets, dim=-1) >= 1e-10).float()  # source-wise l2
    l2_residual = (l2_square(mix_signal - targets, dim=-1) >= 1e-10).float()  # residual l2
    nan_mat = l2_sources * l2_residual  # either or, we discard result if its silent
    nan_mat[nan_mat == 0] = torch.nan  # assign nan to all invalid sources
    return nan_mat


def sqrt_hann(window_length, normalized=False):
    """Implement a sqrt-Hann window"""
    window = torch.sqrt(torch.hann_window(window_length, periodic=True))
    if normalized:
        window = window / torch.sum(window)
    return window


def stft(x: torch.Tensor, fft_size=1024, hop_length=256, window_length=1024, normalize=True):
    if x.ndim == 2:
        x = x.unsqueeze(1)  # add dummy dimension if no sources
    nbatch, nsrcs, ntimes = x.shape
    x = x.reshape(nbatch * nsrcs, ntimes)

    window = sqrt_hann(window_length, normalized=normalize).to(x)

    X = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        normalized=False,
        center=True,
        return_complex=True,
    )
    X = X.permute(0, 2, 1)  # [nbatch * nsrcs, nframes, nfreqs]
    _, nframes, nfreqs = X.shape
    return X.reshape(nbatch, nsrcs, nframes, nfreqs).squeeze(1)


def istft(X: torch.Tensor, fft_size=1024, hop_length=256, window_length=1024, normalize=True, signal_length=None):
    if X.ndim == 3:
        X = X.unsqueeze(-1)  # add dummy dimension if no sources
    nbatch, nframes, nfreqs, nsrcs = X.shape
    X = X.permute(0, 3, 2, 1)
    X = X.reshape(nbatch * nsrcs, nfreqs, nframes)

    window = sqrt_hann(window_length, normalized=normalize).to(X.real)

    x = torch.istft(
        X,
        fft_size,
        hop_length=hop_length,
        win_length=window_length,
        length=signal_length,
        window=window,
        center=True,
        normalized=False,
    )

    x = x.reshape(nbatch, nsrcs, -1).squeeze(1)
    return x


def mask_to_waveform(masks, mix_stft, target_length=None, istft_kwargs=None):
    src_stft = mix_stft.unsqueeze(-1) * masks
    waveform = istft(src_stft, signal_length=target_length, **istft_kwargs)
    return waveform


def log_mag(mix_stft):
    return torch.log(torch.abs(mix_stft) + EPSILON)


def dpcl_mag_weights(mix_stft):
    n_batch, n_time, n_freq = mix_stft.shape
    mag = torch.abs(mix_stft)
    return (mag / torch.sum(mag, (1, 2), keepdim=True)) * n_time * n_freq
