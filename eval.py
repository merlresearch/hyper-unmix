# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from lsx_dataset import LSXDataset
from separate import DEFAULT_PRE_TRAINED_MODEL_PATH, read_checkpoint


def _lightning_eval():
    parser = ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="The path to the LSX directory containing ``tr``, ``cv``,  and ``tt`` directories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_PRE_TRAINED_MODEL_PATH,
        help="Path to trained model weights. Can be a pytorch_lightning checkpoint or pytorch state_dict",
    )
    parser.add_argument("--gpu-device", default=-1, type=int, help="The gpu device for model inference. (default: -1)")
    args = parser.parse_args()

    test_dataset = LSXDataset(args.root_dir, "tt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        drop_last=False,
    )

    if args.gpu_device >= 0:
        devices = [args.gpu_device]
        accelerator = "gpu"
    else:
        devices = "auto"
        accelerator = "cpu"

    trainer = Trainer(
        devices=devices,
        accelerator=accelerator,
        enable_progress_bar=True,  # this will print the results to the command line
        limit_test_batches=1.0,
    )

    model = read_checkpoint(args.checkpoint)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    _lightning_eval()
