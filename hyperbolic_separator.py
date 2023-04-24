# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from argparse import Namespace
from copy import deepcopy

import torch
from geoopt import optim as geo_optim
from pytorch_lightning import LightningModule

from hyperbolic.audio_utils import find_invalid_sources, log_mag, mask_to_waveform, si_snr, stft
from hyperbolic.model.loss import HierarchicalLoss
from hyperbolic.model.model import MaskInference
from lsx_dataset import SOURCE_NAMES_CHILDREN, SOURCE_NAMES_PARENT


class HyperbolicSigSep(LightningModule):
    def __init__(self, hparams: Namespace, model: MaskInference = None):
        """Main hyperbolic audio source separation module

        See [Petermann et al. 2023](https://arxiv.org/pdf/2212.05008.pdf)

        Args:
            hparams (Namespace): the  experiment hyperparameters converted as a Namespace object
            model (MaskInference, optional): the separation model to use. Defaults to None.
        """

        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = model
        if self.model is None:
            self.model = MaskInference(**self.hparams.model)

        self.loss = HierarchicalLoss(stft_kwargs=self.hparams.features, **self.hparams.loss)

        self.stft_kwargs = self.hparams.features
        self.num_parent = len(SOURCE_NAMES_PARENT)

    def forward(self, mix_signal):
        mix_stft = stft(mix_signal, **self.stft_kwargs)
        inp = log_mag(mix_stft)
        out, _ = self.model(inp)
        return out, mix_stft

    def _mlr_to_waveform(self, estimates, mix_stft, signal_length=None):
        est_parents, est_children = torch.tensor_split(estimates, [self.num_parent], -1)
        est_parents, est_children = torch.softmax(est_parents, -1), torch.softmax(est_children, -1)
        return mask_to_waveform(
            torch.concat([est_parents, est_children], -1),
            mix_stft,
            target_length=signal_length,
            istft_kwargs=self.stft_kwargs,
        )

    def separate(self, mix_signal):
        n_batch, n_samples = mix_signal.shape
        estimates, mix_stft = self.forward(mix_signal)
        return self._mlr_to_waveform(estimates, mix_stft, signal_length=n_samples)

    def _step(self, batch, batch_idx, split):
        mix_signal, stacked_parents, stacked_children, track_name = batch

        estimates, mix_stft = self.forward(mix_signal)
        est_parents, est_children = torch.tensor_split(estimates, [self.num_parent], -1)
        loss = self.loss(est_parents, est_children, stacked_parents, stacked_children, mix_stft)

        self.log(f"{split}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        mix_signal, stacked_parents, stacked_children, track_name = batch

        wf_estimates = self.separate(mix_signal)
        nbatch, nsrcs, nsamples = wf_estimates.shape
        wf_targets = torch.concatenate([stacked_parents, stacked_children], 1)

        # check for silent sources/residual. source is expected to be dim=1
        mix_signal = mix_signal.unsqueeze(1).repeat(1, nsrcs, 1)  # expand to nsrcs for residual
        nan_mat = find_invalid_sources(mix_signal, wf_targets)

        # est sisnr
        est_sdr = -si_snr(wf_estimates, wf_targets, dim=-1)
        est_sdr = (est_sdr * nan_mat).nanmean(0)  # average of batch

        # # noisy sisnr
        noisy_sdr = -si_snr(mix_signal, wf_targets, dim=-1)
        noisy_sdr = (noisy_sdr * nan_mat).nanmean(0)  # average of batch

        est_sdr, noisy_sdr = torch.tensor(est_sdr), torch.tensor(noisy_sdr)

        result_dict, srcs_groups = {}, deepcopy(SOURCE_NAMES_CHILDREN)
        srcs_groups.insert(0, SOURCE_NAMES_PARENT)
        all_srcs = [src for group in srcs_groups for src in group]
        for i, src in enumerate(all_srcs):
            if noisy_sdr[i] == noisy_sdr[i]:
                result_dict[f"noisy_{src}"] = noisy_sdr[i].item()
            if est_sdr[i] == est_sdr[i]:
                result_dict[f"est_{src}"] = est_sdr[i].item()
            result_dict["global_avg"] = torch.nanmean(est_sdr).item()
        self.log_dict(result_dict, on_epoch=True)

        return est_sdr.detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = geo_optim.RiemannianAdam(self.parameters(), lr=self.hparams.training["learning_rate"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        lr_scheduler_config = {"scheduler": lr_scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}
        return [optimizer], [lr_scheduler_config]
