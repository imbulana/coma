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
conda create -n coma python=3.11
conda activate coma
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

- Tokenizer: Uses miditok REMI tokenizer, either loaded, untrained, or trained from scratch on the training set to a target vocab size.

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

## Results (wip)

Preliminary results (top K by number of compositions, 80:20 shuffled split, <10 epochs):

| # composers                          | Composition F1 (Confidence Agg) | Composition F1 (Majority Vote) | Chunk F1 | # params |
|--------------------------------------|---------------------------------|--------------------------------|----------|----------|
| 3 [(config)](/configs/K=3.json)      | 0.98                            | 0.98                           | 0.84     | 406,948  |
| 5 [(config)](/configs/K=5.json)      | 0.95                            | 0.92                           | 0.82     | 406,372  |
| 10 [(config)](/configs/K=10.json)    | 0.89                            | 0.84                           | 0.67     | 407,812  | 

## Related Works

[`Deep Composer Classification Using Symbolic Representation (2020)`](https://arxiv.org/pdf/2010.00823)

[`Visual-based Musical Data Representation for Composer Classification (2022)`](https://doi.org/10.1109/iSAI-NLP56921.2022.9960254)

[`ComposeInStyle: Music composition with and without Style Transfer (2021)`](https://doi.org/10.1016/j.eswa.2021.116195)

[`Composer Classification with Cross-modal Transfer Learning and Musically-informed Augmentation (2021)`](https://archives.ismir.net/ismir2021/paper/000100.pdf) (zero-shot)

[`Automated Thematic Composer Classification Using Segment Retrieval (2024)`](https://doi.org/10.1109/MIPR62202.2024.00032)

The following work achieves perfect acc/f1. Based on their code release, it appears that there is likely data leakage. 
Will verify ltr : ) 

[`NLP-based music processing for composer classification (2023)`](https://doi.org/10.1038/s41598-023-40332-0)

code: https://github.com/SirawitC/NLP-based-music-processing-for-composer-classification

## References

This repo is largely adapted from the following.

local attention: https://github.com/lucidrains/local-attention

conformer: https://github.com/jreremy/conformer, https://github.com/lucidrains/conformer

miditok: https://github.com/Natooz/MidiTok

## Citations

```bibtex
@misc{
    gulati2020conformer,
    title   = {Conformer: Convolution-augmented Transformer for Speech Recognition},
    author  = {Anmol Gulati and James Qin and Chung-Cheng Chiu and Niki Parmar and Yu Zhang and Jiahui Yu and Wei Han and Shibo Wang and Zhengdong Zhang and Yonghui Wu and Ruoming Pang},
    year    = {2020},
    eprint  = {2005.08100},
    archivePrefix = {arXiv},
    primaryClass = {eess.AS}
}
```

```bibtex
@article{
    Cui_Hu_Huang_2025, 
    title={Music sequence generation and arrangement based on Transformer model}, 
    author={Cui, Xiao Hong and Hu, Pan and Huang, Zheng}, 
    journal={Journal of Computational Methods in Sciences and Engineering}, 
    year={2025}
    DOI={10.1177/14727978251337904}, 
} 
```

```bibtex
@inproceedings{
  hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}
```

[^1]: [`Conformer: Convolution-augmented Transformer for Speech Recognition`](https://arxiv.org/pdf/2005.08100)

[^2]: [`Music sequence generation and arrangement based on transformer model`](https://doi.org/10.1177/14727978251337904)

[^3]: [`The MAESTRO Dataset`](https://magenta.tensorflow.org/datasets/maestro)