# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple, Union

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from hyperbolic_separator import HyperbolicSigSep
from lsx_dataset import LSXDataset


def _get_dataloaders(
    root_dir: Union[str, Path],
    train_batch_size: int = 25,
    train_chunk_sec: float = 3.2,
    eval_batch_size: int = 5,
    num_workers: int = 4,
) -> Tuple[DataLoader]:

    """
    Returns dataloaders for each of the LSX subsets

    :param root_dir (Union[str, Path]): the path to the LSX directory containing ``tr`` ``cv``  and ``tt`` directories
    :param train_batch_size (int): batch size used for training
    :param train_chunk_sec (float): the training chunk size in seconds
    :param eval_batch_size (int): batch size used for validation
    :param num_workers (int): the number of workers to use by the dataloaders
    :return: (tuple, torch.utils.data.Dataloader): Dataloader objects for each of the three subsets,
             as a tuple (tr, cv, tt)
    """

    train_dataset = LSXDataset(root_dir, "tr", chunk_size_sec=train_chunk_sec, random_start=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    valid_dataset = LSXDataset(root_dir, "cv")
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_dataset = LSXDataset(root_dir, "tt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, valid_loader, test_loader


def cli_main():

    parser = ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="The path to the LSX directory containing ``tr`` ``cv``  and ``tt`` directories.",
    )
    parser.add_argument(
        "--conf-dir",
        default=Path("./hyperbolic/model/conf.yaml"),
        type=Path,
        help="The directory where the YAML configuration file is stored (conf.yaml).",
    )
    parser.add_argument(
        "--exp-dir", default=Path("./exp"), type=Path, help="The directory to save checkpoints and logs."
    )
    args = parser.parse_args()

    with open(args.conf_dir, "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        hparams = Namespace(**params)

    if hparams.training["seed"] is None:
        hparams.training["seed"] = random.getrandbits(16)
    seed_everything(hparams.training["seed"])

    model = HyperbolicSigSep(hparams=hparams)
    train_loader, valid_loader, test_loader = _get_dataloaders(root_dir=args.root_dir, **hparams.data)

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=5, verbose=True)
    callbacks = [checkpoint]
    if hparams.training["num_gpu"] > 0:
        devices = hparams.training["num_gpu"]
        accelerator = "gpu"
    else:
        devices = "auto"
        accelerator = "cpu"

    trainer = Trainer(
        max_epochs=hparams.training["epochs"],
        default_root_dir=args.exp_dir,
        devices=devices,
        accelerator=accelerator,
        gradient_clip_val=5.0,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, valid_loader)
    model.load_from_checkpoint(checkpoint.best_model_path)
    ckpt = torch.load(checkpoint.best_model_path, map_location="cpu")
    torch.save(ckpt, Path(checkpoint.dirpath) / "best_model.ckpt")
    trainer.test(model, test_loader)


if __name__ == "__main__":
    cli_main()
