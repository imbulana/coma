Conformer with multi-scale local attention and *(periodic positional encoding) for composer classification.

## Todo

- [ ] periodic positional encoding (https://doi.org/10.1177/14727978251337904).

## Setup

1. Create a conda environment with python 3.11:

```bash
conda create -n mape python=3.11
conda activate mape
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download the Maestro 3.0 dataset

```bash
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip 'maestro-v3.0.0-midi.zip'
rm 'maestro-v3.0.0-midi.zip'
mv 'maestro-v3.0.0' 'data/maestro-v3.0.0'
```

## Usage

Adjust training params in [`config.py`](/config.py) and begin training the transformer with

```bash
python3 train.py
```

Tensorboard logs and plots will be saved in the specified `LOG_DIR` directory. View the logs with

```bash
tensorboard --logdir=logs
```

## References

This repo is largely adapted from the following.

local attention: https://github.com/lucidrains/local-attention

conformer: https://github.com/jreremy/conformer, https://github.com/lucidrains/conformer

## Citations