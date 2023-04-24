# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import librosa
import librosa.display
import numpy as np
import seaborn as sns
import soundfile as sf
import torch
from scipy.signal import resample_poly

from hyperbolic import audio_utils as au
from hyperbolic.hypertools.hypernn import certainty_from_coors

SR = 16000


class Processor:
    def __init__(self, logger, model, stft_kwargs):
        super().__init__()
        self.model = model
        self.audio = None
        self.input_features = None
        self.mesh_spec = None
        self.mix_stft = None
        self.stft_kwargs = stft_kwargs
        self.projection = None
        self.db_threshold = -5.0
        self.cert_threshold = 0.0
        self.cert_mask = None
        self.logger = logger
        self.max_sec = 8.0

        self.projected_idxs = []

    def set_audio(self, path):
        audio, sr = sf.read(path, dtype=np.float32, always_2d=True)

        # resample
        if not sr == SR:
            audio = resample_poly(audio, SR // 100, sr // 100)

        # mono
        audio = np.mean(audio, 1, keepdims=True)

        if len(audio) > (self.max_sec * SR):
            self.logger.add_log(f"Warning: trimming audio to {self.max_sec}")
            audio = audio[: int(self.max_sec * SR)]

        self.audio = audio
        self.logger.add_log("Audio set from path: {}, with shape: {}".format(path, self.audio.shape))
        self.prepare_audio()
        self.set_mesh_spec()

    def set_mesh_spec(self):
        D = np.abs(self.mix_stft.numpy())
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        S = librosa.display.specshow(S_db)
        self.mesh_spec = S.get_array().reshape(self.mix_stft.numpy().shape)

    def get_mesh_spec(self):
        return self.mesh_spec

    def prepare_audio(self):
        audio_torch = torch.from_numpy(self.audio).T
        self.mix_stft = au.stft(audio_torch, **self.stft_kwargs)
        self.input_features = au.log_mag(self.mix_stft)
        self.mix_stft = self.mix_stft.squeeze(0)
        self.logger.add_log("Audio prepared with shape: {}".format(self.input_features.size()))

    def project_audio(self, path=None):
        if self.input_features is None:
            self.logger.add_log("No Audio Loaded, you need to load an audio file first")
            return

        self.logger.add_log("Inferring from audio features...")
        masks, pc_out = self.model(self.input_features)
        self.projection = pc_out
        self.srcs_masks = masks
        self.logger.add_log("Inferred data with shape: {}".format(pc_out.size()))

    def get_data_points_as_xy(self):
        data_dict = {"x": [], "y": [], "cs": [], "visible": []}
        n_batch, n_frames, n_freqs = self.input_features.shape

        # get color palette
        c = (np.array(sns.color_palette("YlOrRd_r", n_colors=2 * n_freqs, as_cmap=False))[:n_freqs] * 255).astype(int)
        c = np.tile(c.reshape(1, 1, n_freqs, 3), (n_batch, n_frames, 1, 1))

        # to numpy and remove batch dim
        data = self.projection.cpu().detach().numpy().reshape(-1, 2)
        visible_tags = np.zeros(data.shape[0]).astype(bool)

        # filter out-of-db threshold bins
        mx_idx = np.where(self.input_features.clone().detach().cpu().numpy().flatten() >= self.db_threshold)[0]
        # filter out out-of-certainty threshold bins
        cert_idx = np.where(certainty_from_coors(data) >= self.cert_threshold)[0]
        self.cert_mask = np.zeros(n_frames * n_freqs)
        self.cert_mask[cert_idx] = 1.0

        # take intersection of both
        f_idx = np.intersect1d(mx_idx, cert_idx)
        visible_tags[f_idx] = True

        # Each data point holds a color (pretty heavy)
        colors = c.reshape(-1, 3)  # [f_idx,:]
        data_dict["x"], data_dict["y"], data_dict["cs"], data_dict["visible"] = (
            data[:, 0],
            data[:, 1],
            colors,
            visible_tags,
        )

        self.projected_idxs = np.arange(data.shape[0])
        return data_dict

    def set_db_threshold(self, t):
        self.db_threshold = t

    def set_cert_threshold(self, t):
        self.cert_threshold = t

    def get_model_geodesics(self):
        p_k = self.model.mask_layer.mlr.p_k.detach().cpu()
        a_k = self.model.mask_layer.mlr.a_k.detach().cpu()
        return p_k, a_k

    def get_mask_from_selection(self, selection):
        selected = self.projected_idxs[selection.astype(int)]
        mask = torch.zeros(self.mix_stft.flatten().shape)
        mask[selected] = 1.0
        if self.cert_threshold != 0.0 and self.cert_mask is not None:
            mask *= self.cert_mask
        mask = torch.reshape(mask, self.mix_stft.shape).unsqueeze(-1)
        return mask

    def synthesize_selection(self, selection, to_disk=False, path=None):

        mask = self.get_mask_from_selection(selection)
        masked_mask = (mask * self.srcs_masks).squeeze(0)

        # switch to wf-domain
        src_stft = self.mix_stft.unsqueeze(-1) * masked_mask
        waveform = au.istft(src_stft.unsqueeze(0), **self.stft_kwargs)
        wf_np = waveform.mean(-2).squeeze(0).detach().cpu().numpy()

        if to_disk:
            sf.write(path, wf_np, samplerate=SR)
            self.logger.add_log("Done synthesizing selection")

        return wf_np
