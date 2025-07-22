Conformer with multi-scale local attention and *(periodic positional encoding) for composer classification.

Model Architecture (see [`src/transformer.py`](src/transformer.py)):

- Embedding: REMI token embedding + scaled sinusoidal positional embedding.

- Encoder: Stack of conformer-like blocks[^1] (FeedForward → Multi-Scale Local Attention → Convolution → FeedForward, with LayerNorm and residuals).
    - Attention: Multi-scale local self-attention[^2] (windowed, not full sequence). Scales aggregated via a weighted sum (learnable weight for each scale).
    - Convolution: Depthwise 1D convolution.

- Sequence Attention: After encoding, a linear layer computes attention weights over the sequence, producing a weighted sum (sequence embedding).

- Classifier: MLP (LayerNorm → Dropout → Linear → GELU → Dropout → Linear) to output logits for composer classes.

## Todo

- [ ] *periodic positional encoding[^2].

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

Download the Maestro 3.0 dataset[^3]

```bash
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip 'maestro-v3.0.0-midi.zip'
rm 'maestro-v3.0.0-midi.zip'
mv 'maestro-v3.0.0' 'data/maestro-v3.0.0'
```

Data Split & Preprocessing:

There are various options for data preparation and splitting:

- Tokenizer: Uses miditok REMI tokenizer, either loaded or trained from scratch on the training set.

- Select Composers: Only top K composers (by number of compositions or total duration) are selected (`TOP_K_COMPOSERS` in config).

- Train/Test: For each composer, compositions are split so that no composition appears in more than one split (ensures no data leakage). If `SHUFFLE=False`, the data split provided in the `MAESTRO` dataset is used.

- Shuffle: Optionally shuffles before splitting (maintaining that no composition appears in more than one split). This creates a stratified split based on `TEST_SIZE` in config. For consisency with current literature it is recommeded to shuffle the dataset.

- Augmentation: Optionally applies pitch, velocity, and duration augmentations to training data.

## Training

Adjust training params in [`config.py`](/config.py) and begin training the transformer with

```bash
python3 train.py
```

Tensorboard logs and eval plots will be saved in the specified `LOG_DIR` directory. View the logs with

```bash
tensorboard --logdir=<LOG_DIR>
```

Training Details:

- Loss: Cross-entropy loss for multi-class classification.

- Optimizer: AdamW. Configure learning rate and weight decay in config.

- Metrics: Tracks accuracy and F1-score, both at chunk and composition level (majority voting or confidence aggregation).

## Results (todo)

Preliminary results (top K by number of compositions, 80:20 shuffled split, 9 epochs):

| K | Composition F1 (Confidence) | Composition F1 (Majority) | Chunk F1 |
|---|-----------------------------|---------------------------|----------|
| 3 | 90.59                       | 90.15                     | 69.11    |
| 5 | 84.79                       | 85.26                     | 70.91    |

## Related Works

[`Deep Composer Classification Using Symbolic Representation (2020)`](https://arxiv.org/pdf/2010.00823)

[`Visual-based Musical Data Representation for Composer Classification (2022)`](https://doi.org/10.1109/iSAI-NLP56921.2022.9960254)

[`ComposeInStyle: Music composition with and without Style Transfer` (2021)](https://doi.org/10.1016/j.eswa.2021.116195)

[`Composer Classification with Cross-modal Transfer Learning and Musically-informed Augmentation (2021)`](https://archives.ismir.net/ismir2021/paper/000100.pdf) (zero-shot)

The following works achieve perfect acc/f1 but it is unclear and not explicitly mentioned if and how they ensure that no composition (by title) is in more than one split.

[`NLP-based music processing for composer classification (2023)`](https://doi.org/10.1038/s41598-023-40332-0)

[`Automated Thematic Composer Classification Using Segment Retrieval (2024)`](https://doi.org/10.1109/MIPR62202.2024.00032)

## References

This repo is largely adapted from the following.

local attention: https://github.com/lucidrains/local-attention

conformer: https://github.com/jreremy/conformer, https://github.com/lucidrains/conformer

miditok: https://github.com/Natooz/MidiTok

## Citations

[^1]: [`Conformer: Convolution-augmented Transformer for Speech Recognition`](https://arxiv.org/pdf/2005.08100)

[^2]: [`Music sequence generation and arrangement based on transformer model`](https://doi.org/10.1177/14727978251337904)

[^3] [`The MAESTRO Dataset`](https://magenta.tensorflow.org/datasets/maestro)