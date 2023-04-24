# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.nn as nn
from torch.nn import functional as F

from hyperbolic.audio_utils import dpcl_mag_weights, mask_to_waveform, stft
from hyperbolic.model.mask_targets import mask_approximation


class HierarchicalLoss(torch.nn.Module):
    def __init__(self, eps=1e-20, mag_weights=True, loss_type="mask", stft_kwargs=None, **kwargs):
        """Hierarchical loss module

        See [Petermann et al. 2023](https://arxiv.org/pdf/2212.05008.pdf)

        Args:
            eps (float, optional): epsilon value for stability. Defaults to 1e-20.
            mag_weights (bool, optional): whether to weigh the tf bins based on their magnitude . Defaults to True.
            loss_type (str, optional): the type of loss to compute. Defaults to "mask".
                'mask': l1 or nll on mask
                'spectrogram': l1 on spectrogram
                'waveform': l1 on waveform
            stft_kwargs (dict, optional): dictionary of arguments passed to the stft function. Defaults to None.
        """
        super(HierarchicalLoss, self).__init__()

        self.loss_type = loss_type
        self.stft_kwargs = stft_kwargs
        self.mag_weights = mag_weights
        self.eps = eps

        if loss_type == "mask":
            self.mask_type = "ibm"
        elif loss_type == "spectrogram":
            self.mask_type = "psa"

        if loss_type in ["waveform", "spectrogram"]:
            self.loss = self._get_loss("l1")
        elif loss_type == "mask":
            self.loss = self._get_loss("nll")

    def _mask_applier(self, mask_estimates, mix_spectrograms):
        return mix_spectrograms.unsqueeze(-1).expand_as(mask_estimates) * mask_estimates

    def _get_loss(self, loss_str, reduction="none"):

        if loss_str == "l1":
            return nn.L1Loss(reduction=reduction)
        elif loss_str == "nll":
            return nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise ValueError("Parameter loss_type is expected to be a string (choose from nll / l1)")

    def forward(self, est_parents, est_children, target_parents, target_children, mix_stft):
        """
        Compute the hierarchical loss given real-valued estimate, target masks, and the complex input mixture.
        Args:
            est_parents (torch.Tensor): real-valued 3D Tensor with shape [batch, frames, bins, NUM_PARENTS]
            est_children (torch.Tensor): real-valued 3D Tensor with shape [batch, frames, bins, NUM_CHILDREN]
            target_parents (torch.Tensor): real-valued 3D Tensor with shape [batch, frames, bins, NUM_PARENTS]
            target_children (torch.Tensor): real-valued 3D Tensor with shape [batch, frames, bins, NUM_CHILDREN]
            mix_stft (torch.Tensor): complex-valued 3D Tensor with shape [batch, frames, bins]
        Returns:
            float: reduced hierarchical loss
        """
        if self.loss_type == "mask":
            loss = self.maskinf_based_loss(est_parents, est_children, target_parents, target_children, mix_stft)
        elif self.loss_type == "waveform":
            loss = self.waveform_based_loss(est_parents, est_children, target_parents, target_children, mix_stft)
        elif self.loss_type == "spectrogram":
            loss = self.spectrogram_based_loss(est_parents, est_children, target_parents, target_children, mix_stft)

        return loss

    def _get_mask_from_wf(self, x, mix_stft, source_dim=1):
        X = stft(x, **self.stft_kwargs)  # (nbatch, nsrcs, nframes, nfreqs)
        return mask_approximation(
            mix_stft, X, mask_type=self.mask_type, source_dim=source_dim
        )  # (nbatch, nframes, nfreqs, nsrcs)

    def maskinf_based_loss(
        self,
        est_parents: torch.Tensor,
        est_children: torch.Tensor,
        wf_parents: torch.Tensor,
        wf_children: torch.Tensor,
        mix_stft,
        **kwargs
    ):

        n_batch, n_pad_frames, n_freq_bins, _ = est_parents.shape
        n_srcs = est_parents.shape[-1] + est_children.shape[-1]
        normalizer = n_batch * n_pad_frames * n_freq_bins * n_srcs
        weights = dpcl_mag_weights(mix_stft + self.eps) if self.mag_weights else 1.0

        target_p = self._get_mask_from_wf(wf_parents, mix_stft, source_dim=1)  # (nbatch, nframes, nfreqs, nsrcs)
        target_c = self._get_mask_from_wf(wf_children, mix_stft, source_dim=1)

        # compound loss over all individual hierarchy levels (currently limited to parents and children)
        loss_mask = (
            self.loss(est_parents.permute(0, 3, 1, 2), torch.argmax(target_p.permute(0, 3, 1, 2), dim=1)) * weights
        )
        loss_mask += (
            self.loss(est_children.permute(0, 3, 1, 2), torch.argmax(target_c.permute(0, 3, 1, 2), dim=1)) * weights
        )

        loss = torch.sum(loss_mask) / normalizer

        return loss

    def spectrogram_based_loss(
        self,
        est_parents: torch.Tensor,
        est_children: torch.Tensor,
        wf_parents: torch.Tensor,
        wf_children: torch.Tensor,
        mix_stft,
        **kwargs
    ):

        n_batch, n_pad_frames, n_freq_bins, _ = est_parents.shape
        n_srcs = est_parents.shape[-1] + est_children.shape[-1]
        normalizer = n_batch * n_pad_frames * n_freq_bins * n_srcs

        est_mask_parents, est_mask_children = F.softmax(est_parents, dim=-1), F.softmax(est_children, dim=-1)
        est_parents_specs = self._mask_applier(est_mask_parents, mix_stft)
        est_children_specs = self._mask_applier(est_mask_children, mix_stft)
        est_specs = torch.concat([est_parents_specs, est_children_specs], -1)

        target_p = self._get_mask_from_wf(wf_parents, mix_stft, source_dim=1)  # (nbatch, nsrcs, nframes, nfreqs)
        target_c = self._get_mask_from_wf(wf_children, mix_stft, source_dim=1)

        target_specs = torch.concat([target_p, target_c], 1).permute(0, 2, 3, 1)
        loss_spec = torch.sum(self.loss(est_specs, target_specs), dim=-1)

        loss = torch.sum(loss_spec) / normalizer

        return loss

    def waveform_based_loss(
        self,
        est_parents: torch.Tensor,
        est_children: torch.Tensor,
        wf_parents: torch.Tensor,
        wf_children: torch.Tensor,
        mix_stft,
    ):

        n_batch, _, n_times = wf_parents.shape
        n_srcs = est_parents.shape[-1] + est_children.shape[-1]
        normalizer = n_batch * n_times * n_srcs

        est_mask_parents, est_mask_children = F.softmax(est_parents, dim=-1), F.softmax(est_children, dim=-1)
        wf_est_p = mask_to_waveform(
            est_mask_parents, mix_stft, target_length=wf_parents.shape[-1], istft_kwargs=self.stft_kwargs
        )
        wf_est_c = mask_to_waveform(
            est_mask_children, mix_stft, target_length=wf_parents.shape[-1], istft_kwargs=self.stft_kwargs
        )

        wf_estimates = torch.concat([wf_est_p, wf_est_c], -2)
        wf_targets = torch.concat([wf_parents, wf_children], -2)

        loss_wf = self.loss(wf_estimates, wf_targets)

        loss = torch.sum(loss_wf) / normalizer

        return loss
