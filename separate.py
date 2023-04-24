# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import torch
import torchaudio

from hyperbolic_separator import HyperbolicSigSep
from lsx_dataset import EXT, SAMPLE_RATE, SOURCE_NAMES_CHILDREN, SOURCE_NAMES_PARENT

DEFAULT_PRE_TRAINED_MODEL_PATH = Path("checkpoints") / "model.ckpt"


def read_checkpoint(checkpoint_path):
    model = HyperbolicSigSep.load_from_checkpoint(checkpoint_path)
    return model.eval()


def _hss_output_to_dict(output):
    """
    Converts HyperbolicSigSep() output to dictionary with one key per output source.

    :param output (torch.tensor): 3D Tensor of shape [nsrcs, channels, samples]
    :return: (dictionary): {src_name: x_samples} where each of the x_samples are 2D Tensor of shape [channels, samples]
    """
    output_dict = {}
    srcs_groups = deepcopy(SOURCE_NAMES_CHILDREN)
    srcs_groups.insert(0, SOURCE_NAMES_PARENT)
    all_srcs = [src for group in srcs_groups for src in group]
    for i, src_name in enumerate(all_srcs):
        output_dict[src_name] = output[i].float()  # model operate in float64 for hyperbolic operations
    return output_dict


def separate_audio(audio_tensor, model_path=DEFAULT_PRE_TRAINED_MODEL_PATH, device=None):
    """
    Separates a torch.Tensor into three stems. If a separation_model is provided, it will be used,
    otherwise the included pre-trained weights will be used.

    :param audio_tensor (torch.tensor): 2D Tensor of shape [channels, samples]
    :param model_path (Path, optional): path to the pre-trained .ckpt separation model
                                       (default: DEFAULT_PRE_TRAINED_MODEL_PATH)
    :param device (int, optional): The gpu device for model inference.
    :return: (dictionary): {src_name: x_samples} where each of the x_samples are 2D Tensor of shape [channels, samples]
    """
    separation_model = read_checkpoint(model_path)
    if device is not None:
        separation_model = separation_model.to(device)
        audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        wf_estimates = separation_model.separate(audio_tensor)  # [channels, srcs, samples]
    return _hss_output_to_dict(wf_estimates.permute(1, 0, 2))  # [srcs, channels, samples]


def separate_file(audio_filepath, output_directory, model_path=DEFAULT_PRE_TRAINED_MODEL_PATH, device=None):
    """
    Takes the path to a wav file, separates it, and saves results as <SOURCE_NAMES_CHILDREN>, <SOURCE_NAMES_PARENT>.
    Wraps separate_audio(). Audio will be resampled if it's not at the correct samplerate.

    :param audio_filepath (Path): path to mixture audio file to be separated
    :param output_directory (Path): directory where separated audio files will be saved
    :param model_path (Path, optional): path to a pre-trained model .ckpt file.
    :param device (int, optional): The gpu device for model inference.
    """
    audio_tensor, fs = torchaudio.load(audio_filepath)
    if fs != SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(audio_tensor, fs, SAMPLE_RATE)
    output_dict = separate_audio(audio_tensor, model_path, device)
    for k, v in output_dict.items():
        output_path = Path(output_directory) / f"{k}{EXT}"
        torchaudio.save(output_path, v.cpu(), SAMPLE_RATE)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to audio file to be hierarchically separated into parent mixes [music, speech] "
        "and children [bass, drums, guitar] and [speech-male, speech-female].",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_PRE_TRAINED_MODEL_PATH,
        help="Path to the model path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./separated_output",
        help="Path to directory for saving output files.",
    )
    parser.add_argument(
        "--gpu-device", default=-1, type=int, help="The gpu device for model inference. (default: -1 [cpu])"
    )
    args = parser.parse_args()
    if args.gpu_device != -1:
        device = torch.device("cuda:" + str(args.gpu_device))
    else:
        device = torch.device("cpu")
    output_dir = args.out_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    separate_file(args.audio_path, output_dir, device=device, model_path=args.model_path)


if __name__ == "__main__":
    cli_main()
