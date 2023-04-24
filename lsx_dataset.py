# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.data as data
import torchaudio

SOURCE_NAMES_PARENT = ["music_mix", "speech_mix"]
SOURCE_NAMES_CHILDREN = [["bass", "drums", "guitar"], ["speech_male", "speech_female"]]
MIXTURE_NAME = "mix"
SAMPLE_RATE = 16000
EXT = ".wav"


class LSXDataset(data.Dataset):
    def __init__(
        self,
        root_path: Union[str, Path],
        subset: str,
        chunk_size_sec: Optional[float] = None,
        random_start: Optional[bool] = False,
    ) -> None:
        """
        :param root_path (Union[str, Path]): path to top level dataset directory
        :param subset (str): Options: [``"tr"``, ``"cv"``, ``"tt"``].
        :param chunk_size_sec (float, optional): in seconds, instead of reading entire file, read only a chunk of
                                                 this size. Default: None
        :param random_start (bool, optional): If True and chunk_size_sec is specified, __get_item()___ will use
                                              a random start sample. Default: False
        """
        self.path = os.path.join(root_path, subset)
        if not os.path.isdir(self.path):
            raise RuntimeError("Dataset not found. Please check root_path")
        if chunk_size_sec is not None:
            self.chunk_size = int(chunk_size_sec * SAMPLE_RATE)
        else:
            self.chunk_size = -1
        self.random_start = random_start
        self.track_list = self._get_tracklist()

    def _get_tracklist(self) -> List[str]:
        path = Path(self.path)
        names = []
        for root, folders, _ in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith(".") or folders or root == path:
                continue
            name = str(root.relative_to(path))
            names.append(name)
        return sorted(names)

    def _get_audio_path(self, track_name: str, source_name: str) -> Path:
        return Path(self.path) / track_name / f"{source_name}{EXT}"

    def _get_chunk_indices(self, track_name: str) -> Tuple[int, int]:
        mix_path = self._get_audio_path(track_name, MIXTURE_NAME)
        num_frames_total = torchaudio.info(mix_path).num_frames
        start_frame = 0
        num_frames_to_read = self.chunk_size
        if num_frames_total <= self.chunk_size:
            num_frames_to_read = -1
        else:
            if self.random_start:
                start_frame = int(torch.randint(0, num_frames_total - self.chunk_size, (1,)))
        return start_frame, num_frames_to_read

    def _read_audio(self, path: Path, frame_offset: Optional[int] = 0, num_frames: Optional[int] = -1) -> torch.Tensor:
        y, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        assert sr == sr, "audio sampling rate of data does not match requested sampling rate"
        return y.squeeze(0)

    def _load_track(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        track_name = self.track_list[index]
        frame_offset, num_frames = self._get_chunk_indices(track_name)
        mix_path = self._get_audio_path(track_name, MIXTURE_NAME)
        y_mix = self._read_audio(mix_path, frame_offset, num_frames)
        parent_wavs = []
        child_wavs = []
        for parent, children in zip(SOURCE_NAMES_PARENT, SOURCE_NAMES_CHILDREN):
            src_path = self._get_audio_path(track_name, parent)
            y_src = self._read_audio(src_path, frame_offset, num_frames)
            parent_wavs.append(y_src)
            for child in children:
                src_path = self._get_audio_path(track_name, child)
                y_src = self._read_audio(src_path, frame_offset, num_frames)
                child_wavs.append(y_src)
        stacked_parents = torch.stack(parent_wavs)
        stacked_children = torch.stack(child_wavs)
        return y_mix, stacked_parents, stacked_children, track_name

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load the index-th example from the dataset.

        :param index (int): sample index to load
        :return:
            Tuple of the following items;
            torch.Tensor:
                mixture [channels, samples]
            torch.Tensor:
                targets [sources, channels, samples]
            str:
                Dataset filename
        """
        return self._load_track(index)

    def __len__(self) -> int:
        return len(self.track_list)
