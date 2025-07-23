import os
import random
import pandas as pd
import shutil

import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset

from sklearn.model_selection import train_test_split

from src import Transformer
from utils import *

# load config

from config import *

# create log dirs

os.makedirs(LOG_DIR / "cm" / "chunk", exist_ok=True)
os.makedirs(LOG_DIR / "f1" / "chunk", exist_ok=True)
os.makedirs(LOG_DIR / "cm" / "composition", exist_ok=True)
os.makedirs(LOG_DIR / "f1" / "composition", exist_ok=True)

# tensorboard writer

writer = SummaryWriter(log_dir=LOG_DIR)
save_config(writer)

# set seed

torch.manual_seed(SEED)
random.seed(SEED)

# load/train tokenizer and maestro data

if USE_PRETRAINED_TOKENIZER and TOKENIZER_LOAD_PATH.exists():
    print(f"\nloading pretrained tokenizer from {TOKENIZER_LOAD_PATH}\n")
    tokenizer = REMI(params=TOKENIZER_LOAD_PATH)
else:
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)
    print(f"\nusing base tokenizer with vocab size: {len(tokenizer)}\n")

if SPLIT_DATA:

    split_parent_path = MAESTRO_DATA_PATH / "splits"
    df = pd.read_csv(MAESTRO_CSV)

    if SORT_BY == 'compositions':
        # select top k composers by number of compositions
        composer_n_compositions = df.groupby('canonical_composer')['canonical_title'].nunique()
        top_k_composers = composer_n_compositions.sort_values(ascending=False).head(TOP_K_COMPOSERS).index

    elif SORT_BY == 'duration':
        # select top k composers by duration
        composer_duration = df.groupby('canonical_composer')['duration'].sum()
        top_k_composers = composer_duration.sort_values(ascending=False).head(TOP_K_COMPOSERS).index
    else:
        raise ValueError(f"Invalid sort_by: {SORT_BY}. Must be 'compositions' or 'duration'.")

    df = df[df['canonical_composer'].isin(top_k_composers)]

    print(f"\nselected {len(top_k_composers)} composers: {top_k_composers.tolist()}\n")
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
        print(f"\ntraining tokenizer to target vocab size: {VOCAB_SIZE}\n")

        train_paths = df[df['split'] == 'train']['midi_filename'].apply(
            lambda x: str(MAESTRO_DATA_PATH / x)
        ).tolist()

        # train the tokenizer with Byte Pair Encoding (BPE) to build the vocabulary

        tokenizer.train(
            vocab_size=VOCAB_SIZE,
            files_paths=train_paths,
        )
        tokenizer.save(TOKENIZER_SAVE_PATH)

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

# collator pads left to longest sequence length in batch
collator = DataCollator(pad_token_id=tokenizer["PAD_None"])
train_chunk_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
val_chunk_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

midi_paths_valid_comp = group_by_composition(midi_paths_valid)
val_composition_dataset = CompositionDataset(midi_paths_valid_comp, **kwargs_dataset)
val_composition_loader = CompositionDataLoader(val_composition_dataset, collator)

# instantiate encoder-classifier model

model = Transformer(
    dim = DIM,
    vocab_size = len(tokenizer),
    max_seq_len = MAX_SEQ_LEN,
    depth = DEPTH,
    dim_head = DIM_HEAD,
    heads = HEADS,
    ff_mult = FF_MULT,
    attn_window_sizes = ATTN_WINDOW_SIZES,
    conv_expansion_factor = CONV_EXPANSION_FACTOR,
    conv_kernel_size = CONV_KERNEL_SIZE,
    attn_dropout = ATTN_DROPOUT,
    ff_dropout = FF_DROPOUT,
    conv_dropout = CONV_DROPOUT,
    num_classes=len(composers),
    prenorm=PRENORM,
    qk_scale=QK_SCALE,
).to(DEVICE)

print(f"\nmodel size: {sum(p.numel() for p in model.parameters()):,}\n")

# optimizer

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
    valid_acc_maj, valid_f1_maj, valid_acc_conf, valid_f1_conf = validate_composition(
        model, val_composition_loader, DEVICE, epoch, writer, composer_id2name, 
        save_path=LOG_DIR, show_plots=False, eval_type='composition'
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