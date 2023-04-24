# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

EPSILON = 1e-12


def irm(srcs, source_dim=-1, eps=EPSILON):
    """
    Ideal ratio mask
    :param srcs: Complex valued torch.tensor of shape [n_batch, n_time, n_freq, n_srcs]
    :param source_dim: Expected dimension of the sources
    :param eps: epsilon (float) to avoid divide by zero
    :return: Real valued torch.tensor of shape [n_batch, n_time, n_freq, n_srcs]
    """
    all_srcs = torch.sum(torch.abs(srcs) + eps, dim=source_dim, keepdim=True)
    return torch.abs(srcs) / all_srcs


def ibm(srcs, source_dim=-1, eps=EPSILON):
    """
    Ideal Binary Mask

    :param srcs: Complex valued torch.tensor of shape [n_batch, n_time, n_freq, n_srcs]
    :param source_dim: Expected dimension of the sources
    :param eps: epsilon (float) to avoid divide by zero
    :return: Real valued torch.tensor of shape [n_batch, n_time, n_freq, n_srcs]
    """
    masks = irm(srcs, source_dim, eps)
    amax = torch.argmax(masks, dim=source_dim)
    result = torch.nn.functional.one_hot(amax, num_classes=srcs.shape[source_dim])
    return result


def phase_spectrogram_approx(mix, srcs, source_dim=-1, rmax=1):
    """
    Phase-sensitive approximation

    :param mix: Complex valued torch.tensor of shape [n_batch, n_time, n_freq]
    :param srcs: Complex valued torch.tensor of shape [n_batch, n_time, n_freq, n_srcs]
    :param source_dim: Expected dimension of the sources
    :param rmax: Expected dimension of the sources
    :return: Real valued torch.tensor of shape [n_batch, n_time, n_freq, n_srcs]
    """

    mix_stft = torch.unsqueeze(mix, source_dim)
    mix_mag, mix_phase = torch.abs(mix_stft), torch.angle(mix_stft)
    srcs_mag, srcs_phase = torch.abs(srcs), torch.angle(srcs)
    result = torch.minimum(rmax * mix_mag, srcs_mag * torch.cos(srcs_phase - mix_phase))
    result = torch.clamp(result, min=0.0)
    return result


def mask_approximation(
    mix_stft: torch.Tensor, srcs_stft: torch.Tensor, mask_type=None, source_dim=-1, rmax=1.0, eps=EPSILON
):

    if mask_type == "ibm":
        all_masks = ibm(srcs_stft, source_dim, eps)
    elif mask_type == "irm":
        all_masks = irm(srcs_stft, source_dim, eps)
    elif mask_type == "psa":
        all_masks = phase_spectrogram_approx(mix_stft, srcs_stft, source_dim, rmax=rmax)
    else:
        raise ValueError("Unknown mask type: {}!".format(mask_type))

    return all_masks
