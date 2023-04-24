<!--
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Hyperbolic Audio Source Separation

<p align="left">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=RKsAMb9z70Y" target="_blank">
 <img src="/docs/yt_thumbnail.png" alt="Watch the video" width="800"border="4" />
</a>
</p>


:movie_camera: [Please click the image above to watch the demo video.](https://www.youtube.com/watch?v=RKsAMb9z70Y)

:page_facing_up: [Please click here to read the paper.](https://arxiv.org/pdf/2212.05008.pdf)

If you use any part of this code for your work, we ask that you include the following citation:

    @InProceedings{Petermann2023ICASSP_hyper,
      author    =  {Petermann, Darius and Wichern, Gordon and Subramanian, Aswin and {Le Roux}, Jonathan},
      title     =  {Hyperbolic Audio Source Separation},
      booktitle =	 {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year      =	 2023,
      month     =	 jun
    }
***
## Table of contents

1. [Environment Setup](#environment-setup)
2. [Training a model on the LSX Dataset](#training-a-model-on-the-lsx-dataset)
3. [Evaluating a model on the LSX Dataset](#evaluating-a-model-on-the-lsx-dataset)
4. [Exploring the hyperbolic user interface](#exploring-the-hyperbolic-user-interface)
5. [Contributing](#contributing)
6. [License](#license)
***
## Environment Setup

The code has been tested using `python 3.9` on both Linux and macOS. Necessary dependencies can be installed using the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

The pytorch installation includes the [latest version](https://pytorch.org/get-started/pytorch-2.0/), which you may modify based on your local CUDA requirements.

***

## Training a model on the LSX Dataset

In order to evaluate and/or train a model from scratch, download first the [LSX dataset](https://zenodo.org/record/7765140#.ZCio-S-B0lI), which is around 30GB in size. The data is already formatted and no pre-processing is needed.

Experiment configuration is done through the [conf.yaml](hyperbolic/model/conf.yaml) file from which data, training, and model pipelines can be established. While most parameters are self-explanatory and may be left unchanged unless needed, the followings will allow you to explore our hyperbolic framework extensively:

- **model**:
  - `hyperbolic_dim` : the dimension of the hyperbolic embeddings (default: `2`)
  - `hyperbolic_train` : whether or not the Poincaré Ball curvature is trainable (default: `false`)
  - `hyperbolic_k` : Poincaré Ball curvature (default: `1.0`)

- **loss**:
  - `loss_type` : the loss type used during training (either `mask`, `spectrogram`, or `waveform`-based, default: `mask`)

Once the conf.yaml is setup, the training script can be run with the following:

```bash
python lightning_train.py \
        --root-dir LSX_ROOT_DIR \
        [--conf-dir CONF_DIR] \
        [--exp-dir EXP_DIR] \
```

where `LSX_ROOT_DIR` denotes the root path of the LSX dataset (i.e., containing the `tr`, `cv`, and `tt` folders), `CONF_DIR` the folder containing the YAML configuration file (conf.yaml), and `EXP_DIR` the parent folder in which the experiment directory will be created.

***

## Evaluating a model on the LSX Dataset

We provide a [pre-trained model](checkpoints) (trained with `hyperbolic_k = 1.0` and `hyperbolic_dim = 2`), that one can download via `git lfs`.

To evaluate the scale-invariant source to distortion ratio (SI-SDR) on the LSX test set using a pre-trained model, run:

```bash
python eval.py \
        --root-dir LSX_ROOT_DIR \
        [--checkpoint CHECKPOINT] \
        [--gpu-device GPU_DEVICE] \
```

The following is the average (over all classes and examples) SI-SDR (dB) of the LSX test set using various
configurations of parameters from [conf.yaml](hyperbolic/model/conf.yaml)

| loss_type   | hyperbolic_k | hyperbolic_dim | this code | paper |
|:------------|-------------:|---------------:|----------:|------:|
| mask        |          1.0 |              2 |       6.6 |   6.3 |
| spectrogram |          1.0 |              2 |       6.1 |   2.1 |
| waveform    |          1.0 |              2 |       5.6 |   5.4 |
| mask        |          0.0 |              2 |       6.4 |   6.1 |
| mask        |          1.0 |            128 |       6.9 |   6.6 |
| mask        |          0.0 |            128 |       7.0 |   6.8 |

All parameters except those specified in the table were constant at the default values
from [conf.yaml](hyperbolic/model/conf.yaml) on GPUs with 12 GB memory, except for the `hyperbolic_dim=128` rows
where `eval_batch_size=1` and GPUs with 48 GB memory were used.

***
## Exploring the hyperbolic separation user interface

The interface offers an interactive and user-friendly way to explore the hyperbolic space of a pre-trained model by allowing users to load their own audio examples and perform selection-based source separation on the Poincaré Ball.

To get started with the hyperbolic interface, run the main interface script and pass it the ckeckpoint file path you'd like to use:

```bash
python interface.py [--checkpoint PRETRAINED-MODEL-PATH]
```
where `PRETRAINED-MODEL-PATH` denotes a checkpoint path which has been trained on `hyperbolic_dim = 2`. We provide a pre-trained checkpoint, which we discussed in [the evaluation section](#evaluating-a-model-on-the-lsx-dataset).

After running the command, an interactive python window will pop up. While we provide some audio excerpts (which can be found in the [audio folder](interface/audio)) directly taken from the [LSX](https://zenodo.org/record/7765140#.ZCio-S-B0lI) test set, we encourage users to try out with their own material as well. Note that while the interface supports arbitrary file lengths, we recommend file lengths under 10 s for better user experience.

#### GUI Overview

The interface is divided as follows:

<p align="left">
 <img src="/docs/interface.png" alt="Hyperbolic Separation GUI" width="1000" border="0" />
</p>


- __Main Control Panel__: Provides most of the controls needed to interact with the interface.
  - *"Load Audio File"*: As the name suggests. Keep in mind that audio is summed to mono and resampled at 16kHz when loaded. Best results are with audio duration < 8 s. Example files are provided in the [audio folder](interface/audio).
  - *"Scatter Display Threshold (dB)"*: The process of projecting spectrogram bins in hyperbolic space can be rather *slow*. For that matter and especially for longer files, it can be convenient to limit the number of displayed projected embeddings (i.e., T-F bins) based on their magnitude. Note that this *only* controls the visual aspect of the projection; all bins will still be considered during the synthesis process, regardless of this threshold parameter.
  - *"Certainty Synthesis Threshold"*: This more experimental feature allows you to display embeddings based on their certainty (i.e., where they lay on the Poincaré Ball and their distance to the origin). This can be useful when experimenting with certainty and its impact on the perceived audio (i.e., only synthesize embeddings with high certainty).
  - *"Project Audio"*: Will project the hyperbolic embeddings of the loaded audio file onto the Poincaré Ball. The embeddings are obtained by running inference with the pre-trained model and extracting the 2-D feature prior to hyperbolic MLR (i.e., prior to classification). Each T-F bin is represented by a data point on the ball (i.e., the longer the audio, the more data points).
  - *"Class Geodesic"*: Toggle on/off the learned hyperbolic decision boundaries (i.e., Geodesics). When toggled on, a legend will appear, allowing you to select the embeddings located within a single or multiple geodesics. This can be especially useful when making class-based selections (e.g., selecting only drums embeddings).
  - *"Class Intersections"*: Once toggled on, the class-based selection will only take into account class intersections (e.g., all embeddings laying within both the drums and bass geodesics). This can be useful when experimenting with inter-class certainty.

- __Spectrogram Display__: Once an audio file is loaded, its magnitude spectrogram will be displayed on this panel. Moreover, when a selection is performed on the Poincaré Ball, the T-F bins associated with the selected embeddings will be highlighted as a green overlay.

- __Poincaré Ball – Embedding Selection Panel__: This panel gives the ability to perform selection-based audio source separation on the Poincaré Ball. Once an audio file is loaded and its embeddings projected, the user can perform a selection in two ways:
  - **Manual selection**: By dragging the manual selector around the ball area, resizing it (click and drag square handle), and rotating it (click and drag circle handle)
  - **Class-based selection**: By toggling the *"Class Geodesic"* control and making your class selection on the legend, directly.

- __Audio Player Controls__: Once a file has been loaded, the audio can be played back and stopped via this panel (left buttons). Once a selection is made, its audio counterpart will automatically be synthesized; the resulting source-separated output can be played back and stopped using the right buttons.

- __Embedding Statistics Panel__: This panel will display some statistics based on the frequency (i.e., bin height) and hyperbolic certainty (i.e., distance from the origin) of the current Poincaré selection. For example, if the selection is overall located at the very edge of the ball, the certainty distribution will be skewed towards the right. Moreover, we observed in some cases a correlation between T-F bin frequency and their associated certainty: a selection made around the very edge of the ball tends to contain many low-frequency T-F bins while a selection made around the middle will contain higher T-F bins.
We hypothesize that this is because all of the loss functions we use have weights based on T-F bin energy, either directly in the mask/spectrogram case or indirectly in the waveform case, and low frequency bins tend to have more energy, so the network learns to put these bins closer to the edge of the hyperbolic space.

- __Logger Window__: Will display any information going through the interface logging system.

***
## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

***

## Copyright and License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:
```
Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

The following files:

* `hyperbolic/hypertools/hyptorch.py`
* `hyperbolic/hypertools/pmath.py`

were taken without modification from https://github.com/leymir/hyperbolic-image-embeddings (license included in [LICENSES/MIT.md](LICENSES/MIT.md)):

```
Copyright (c) 2019 Valentin Khrulkov
```

The following file:
* `hyperbolic/hypertools/hypernn.py`

was adapted from https://github.com/nlpAThits/hyfi (license included in [LICENSES/MIT.md](LICENSES/MIT.md)):

```
Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2020 HITS NLP
```

The following file:
* `hyperbolic/hypertools/dist2plane.py`

was adapted from https://github.com/geoopt/geoopt (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md)):

```
Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2018 Geoopt Developers
```
