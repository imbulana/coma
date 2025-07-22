# TODO: add argparse, config file
import os
import random
import pandas as pd
import shutil

import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from pathlib import Path

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset

from sklearn.model_selection import train_test_split

from src import Transformer
from utils import *

# config

MAESTRO_DATA_PATH = Path("data/maestro-v3.0.0").resolve()
MAESTRO_CSV = MAESTRO_DATA_PATH / "maestro-v3.0.0.csv"
TOKENIZER_PATH = Path("tokenizer.json").resolve()

# SPLIT_DATA = True if not os.path.exists(MAESTRO_DATA_PATH / "splits") else False
SPLIT_DATA = True
SHUFFLE = True
TRAIN_TOKENIZER = True
TEST_SIZE = 0.2
MIN_COMPOSER_DURATION = 10000
TOP_K_COMPOSERS = 14 # train/test on top k composers
TO_SKIP = []
# TO_SKIP = ['Schubert']
# TO_SKIP = ['Modest', 'Claude', 'Schubert', 'Brahms', 'Haydn', 'Scriabin', 'Mozart', 'Sergei', 'Felix', 'Bach', 'Schumann', 'Mussorgsky']
AUGMENT_DATA = False

LOG_DIR = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOG_DIR / "cm", exist_ok=True)
os.makedirs(LOG_DIR / "f1", exist_ok=True)

SEED = 42
TEST_SIZE = 0.1
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-4
MAX_SEQ_LEN = 1024
DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

writer = SummaryWriter(log_dir=LOG_DIR)

# set seed

torch.manual_seed(SEED)
random.seed(SEED)

# load/train tokenizer and maestro data

if os.path.exists(TOKENIZER_PATH) and not TRAIN_TOKENIZER:
    tokenizer = REMI(params=TOKENIZER_PATH)
else:
    TRAIN_TOKENIZER = True

if SPLIT_DATA:

    split_parent_path = MAESTRO_DATA_PATH / "splits"
    df = pd.read_csv(MAESTRO_CSV)

    # select top k composers by duration

    composer_duration = df.groupby('canonical_composer')['duration'].sum()
    top_k_composers = composer_duration.sort_values(ascending=False).head(TOP_K_COMPOSERS).index

    df= df[df['canonical_composer'].isin(top_k_composers)]

    print(f"\nselected composers: {top_k_composers.tolist()}\n")
    if TO_SKIP:
        print(f"\nskipping composers: {TO_SKIP}\n")

    # shuffle data (ensure that no composition is shared b/w train and valid)

    if SHUFFLE:
        print("\nshuffling data...\n")

        for composer, df_composer in df.groupby('canonical_composer'):

            titles = df_composer['canonical_title'].unique()
            titles_train, titles_test = train_test_split(
                titles, test_size=TEST_SIZE, random_state=SEED, shuffle=True
            )

            df.loc[df['canonical_title'].isin(titles_train), 'split'] = 'train'
            df.loc[df['canonical_title'].isin(titles_test), 'split'] = 'validation'

            print(f"{composer}:")
            print(f"titles_train: {len(titles_train)}, titles_test: {len(titles_test)}\n")

    # train tokenizer

    if TRAIN_TOKENIZER:
        print("\ntraining tokenizer...\n")

        from train_tokenizer import TOKENIZER_PARAMS, VOCAB_SIZE, SAVE_PATH

        train_paths = df[df['split'] == 'train']['midi_filename'].apply(
            lambda x: str(MAESTRO_DATA_PATH / x)
        ).tolist()

        config = TokenizerConfig(**TOKENIZER_PARAMS)
        tokenizer = REMI(config)

        # train the tokenizer with Byte Pair Encoding (BPE) to build the vocabulary

        tokenizer.train(
            vocab_size=VOCAB_SIZE,
            files_paths=train_paths,
        )
        tokenizer.save(SAVE_PATH)

    # remove existing split

    if split_parent_path.exists():
        shutil.rmtree(split_parent_path)
        print(f"\nremoved existing split: {split_parent_path}\n")

    # create new split

    print("\ncreating new split...\n")


    failures = []
    for split, df_split in df.groupby("split"):
        split_path = split_parent_path / split

        for composer, df_composer in df_split.groupby("canonical_composer"):
            if any(skip_composer in str(composer) for skip_composer in TO_SKIP):
                continue

            composer_path = split_path / composer
            midi_file_paths = df_composer["midi_filename"].apply(
                lambda x: MAESTRO_DATA_PATH / x
            ).tolist()
            try:
                for midi_file_path in midi_file_paths:
                    split_files_for_training(
                        files_paths=[midi_file_path],
                        tokenizer=tokenizer,
                        save_dir=composer_path / midi_file_path.name,
                        max_seq_len=MAX_SEQ_LEN,
                        num_overlap_bars=2,
                    )
            except FileNotFoundError:
                failures.append(midi_file_path)

    if failures:
        print(f"\nfailed to split {len(failures)} files")
        print(failures)

leaf = lambda split : f"splits/{split}/*/*/*.mid?"
get_composer_label = lambda dummy1, dummy2, x: x.parent.parent.name # signature expected by DatasetMIDI

midi_paths_train = list(MAESTRO_DATA_PATH.glob(leaf("train")))
midi_paths_valid = list(MAESTRO_DATA_PATH.glob(leaf("validation")))
midi_paths_test = list(MAESTRO_DATA_PATH.glob(leaf("test")))

_SHUFFLE = False
if _SHUFFLE:
    all_midi_paths = midi_paths_train + midi_paths_valid + midi_paths_test
    df_all = pd.DataFrame(all_midi_paths, columns=['path'])
    df_all['composer'] = df_all['path'].apply(lambda x: x.parent.parent.name)

    X_train, X_valid, y_train, y_valid = train_test_split(
        df_all['path'], df_all['composer'], test_size=TEST_SIZE, random_state=42, stratify=df_all['composer']
    )

    midi_paths_train = X_train.tolist()
    midi_paths_valid = X_valid.tolist()

if AUGMENT_DATA:
    augment_dataset(
        MAESTRO_DATA_PATH / "splits" / "train",
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 4],
        duration_offsets=[-0.5, 0.5],
    )

    midi_paths_train = list(MAESTRO_DATA_PATH.glob(leaf("train")))
    print(f"\ntrain samples (augmentations added): {len(midi_paths_train)}")

# create datasets and dataloaders

composers = [path.name for path in (MAESTRO_DATA_PATH / "splits" / "train").glob("*/")]
composer_id2name = {i: composer for i, composer in enumerate(composers)}
composer_name2id = {composer: i for i, composer in composer_id2name.items()}

get_composer_label = lambda dummy1, dummy2, x: composer_name2id[x.parent.parent.name] # signature expected by DatasetMIDI

kwargs_dataset = {
    "max_seq_len": MAX_SEQ_LEN, 
    "tokenizer": tokenizer,
    "bos_token_id": tokenizer["BOS_None"],
    "eos_token_id": tokenizer["EOS_None"],
    "func_to_get_labels": get_composer_label
}

train_dataset = DatasetMIDI(midi_paths_train, **kwargs_dataset)
val_dataset = DatasetMIDI(midi_paths_valid, **kwargs_dataset)

print(f"\ntrain samples: {len(train_dataset)}")
print(f"valid samples: {len(val_dataset)}")
# print(f"test samples: {len(test_dataset)}\n")

collator = DataCollator(pad_token_id=tokenizer["PAD_None"])
train_chunk_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
val_chunk_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

midi_paths_valid_comp = group_by_composition(midi_paths_valid)
val_composition_dataset = CompositionDataset(midi_paths_valid_comp, **kwargs_dataset)
val_composition_loader = CompositionDataLoader(val_composition_dataset, collator)

# instantiate encoder-classifier model

model = Transformer(
    dim = 64,
    vocab_size = len(tokenizer),
    max_seq_len = MAX_SEQ_LEN,
    depth = 1,
    dim_head = 4,
    heads = 4,
    ff_mult = 2,
    attn_window_sizes = [8, 64],
    conv_expansion_factor = 2,
    conv_kernel_size = 31,
    attn_dropout = 0.3,
    ff_dropout = 0.3,
    conv_dropout = 0.3,
    num_classes=len(composers),
    prenorm=True,
    qk_scale=4,
).to(DEVICE)

print(f"\nmodel size: {sum(p.numel() for p in model.parameters()):,}\n")

# optimizer

# optim = Adam(model.parameters(), lr=LEARNING_RATE)
optim = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# training

best_valid_loss = float('inf')
for epoch in range(1, NUM_EPOCHS+1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    
    train_loss, train_acc, train_f1 = train_epoch(
        model, train_chunk_loader, criterion, optim, DEVICE, epoch, writer
    )
    valid_loss, valid_acc, valid_f1 = validate_chunks(
        model, val_chunk_loader, criterion, DEVICE, epoch, writer, composer_id2name, 
        save_path=LOG_DIR, show_plots=False
    )
    # test_loss, test_acc, test_f1 = validate_chunks(
    #     model, test_loader, DEVICE, epoch, writer, composer_id2name, test=True
    # )

    valid_acc_maj, valid_f1_maj, valid_acc_conf, valid_f1_conf = validate_composition(
        model, val_composition_loader, DEVICE, epoch, writer, composer_id2name
    )

    print_metrics(
        train_loss, train_acc, train_f1, 
        valid_loss, valid_acc, valid_f1, 
        valid_acc_maj, valid_f1_maj, 
        valid_acc_conf, valid_f1_conf
    )
    
    # save checkpoint
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f"{LOG_DIR}/best_model.pt")
        print(f"\nmodel saved to '{LOG_DIR}/best_model.pt'\n")

writer.close()