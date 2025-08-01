Conformer with multi-scale local attention and *(periodic positional encoding) for composer classification.

Model Architecture (see [`src/transformer.py`](src/transformer.py)):

- Embedding: REMI token embedding + scaled sinusoidal positional embedding.

- Encoder: Stack of conformer-like blocks[^1] (FeedForward → Multi-Scale Local Attention → Convolution Module → FeedForward, with LayerNorm and residuals).
    - Attention: Multi-scale local self-attention (windowed, not full sequence). Scales aggregated via a weighted sum (learnable weight for each scale). Inspired by the multi-scale attention mechanism in Cui et al.[^2]
    - Convolution Module: pointwise convolution (w/ expansion factor of 2) -> GLU activation -> 1D Depthwise convolution -> Batchnorm -> Swish activation.

- Sequence Attention: After encoding, a linear layer computes attention weights over the sequence, producing a weighted sum (sequence embedding).

- Classifier: MLP (LayerNorm → Linear → GELU → Dropout → Linear) to output logits for composer classes.

## Todo

- [ ] *periodic positional encoding[^2].

## Setup

Create a conda environment with python 3.11:

```bash
conda create -n coma python=3.11
conda activate coma
```

Install dependencies:

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

- Tokenizer: Uses miditok REMI tokenizer, either loaded, untrained, or trained from scratch on the training set to a target vocab size.

- Select Composers: Only top K composers (by number of compositions or total duration) are selected (`TOP_K_COMPOSERS` in config).

- Train/Test splits: For each composer, compositions are split so that no composition appears in more than one split (ensures no data leakage).

- Shuffle (recommended): Optionally shuffles before splitting (maintaining that no composition appears in more than one split). This creates a stratified split based on `TEST_SIZE` in config. If `SHUFFLE=False`, the data split provided in the `MAESTRO` dataset is used.

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

- Optimizer: AdamW.

- LR Scheduler: MultiStepLR or CosineAnnealing.

- Metrics: Tracks accuracy and F1-score, both at chunk and composition level (majority voting or confidence aggregation).

## Results (wip)

Preliminary results (top K by number of compositions, 80:20 shuffled split, 20 epochs):

| # composers                          | Composition F1 (Confidence Agg) | Composition F1 (Majority Vote) | Chunk F1 | # params |
|--------------------------------------|---------------------------------|--------------------------------|----------|----------|
| 3 [(config)](/configs/K=3.json)      | 0.98                            | 0.98                           | 0.84     | 406,948  |
| 5 [(config)](/configs/K=5.json)      | 0.97                            | 0.97                           | 0.86     | 402,921  |
| 10 [(config)](/configs/K=10.json)    | 0.90                            | 0.89                           | 0.69     | 407,812  | 

## Related Works

[`Deep Composer Classification Using Symbolic Representation (2020)`](https://arxiv.org/pdf/2010.00823)[(code)](https://github.com/KimSSung/Deep-Composer-Classification)

[`Visual-based Musical Data Representation for Composer Classification (2022)`](https://doi.org/10.1109/iSAI-NLP56921.2022.9960254)

[`ComposeInStyle: Music composition with and without Style Transfer (2021)`](https://doi.org/10.1016/j.eswa.2021.116195)

[`Composer Classification with Cross-modal Transfer Learning and Musically-informed Augmentation (2021)`](https://archives.ismir.net/ismir2021/paper/000100.pdf) (zero-shot)

[`Automated Thematic Composer Classification Using Segment Retrieval (2024)`](https://doi.org/10.1109/MIPR62202.2024.00032)

[`Concept-Based Explanations For Composer Classification (2022)`](https://archives.ismir.net/ismir2022/paper/000105.pdf)[(code)](https://github.com/CPJKU/composer_concept/tree/main)

The following work achieves perfect acc/f1. Looking at their [code](https://github.com/SirawitC/NLP-based-music-processing-for-composer-classification), it appears that there is data leakage b/w the train and test sets. Their dataset (on which they do a random train/test split) for the 5 composer classification task has at most 482 unique compositions but 809 total compositions.

[`NLP-based music processing for composer classification (2023)`](https://doi.org/10.1038/s41598-023-40332-0)

## References

This repo is largely adapted from the following.

local attention: https://github.com/lucidrains/local-attention

conformer: https://github.com/jreremy/conformer, https://github.com/lucidrains/conformer

miditok: https://github.com/Natooz/MidiTok

[^1]: Gulati, A., Qin, J., Chiu, C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., & Pang, R. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. [ArXiv, abs/2005.08100.](https://arxiv.org/abs/2005.08100)

[^2]: Cui XH, Hu P, Huang Z. Music sequence generation and arrangement based on transformer model. Journal of Computational Methods in Sciences and Engineering. 2025;0(0). [doi:10.1177/14727978251337904.](https://doi.org/10.1177/14727978251337904)

[^3]: Hawthorne, C., Stasyuk, A., Roberts, A., Simon, I., Huang, C.A., Dieleman, S., Elsen, E., Engel, J., & Eck, D. (2018). Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset. [ArXiv, abs/1810.12247.](https://arxiv.org/abs/1810.12247)