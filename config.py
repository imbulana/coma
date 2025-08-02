import torch
from pathlib import Path
from datetime import datetime

SEED = 42
DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

# data paths

MAESTRO_DATA_PATH = Path("data/maestro-v3.0.0").resolve()
MAESTRO_CSV = MAESTRO_DATA_PATH / "maestro-v3.0.0.csv"
# MAESTRO_CSV = MAESTRO_DATA_PATH / "maestro-v3.0.0_standardized.csv"

# data splitting

SPLIT_DATA = True
MAX_SEQ_LEN = 2048
SHUFFLE = True
SORT_BY = 'compositions' # must be in ['compositions', 'duration']
TEST_SIZE = 0.2
TOP_K_COMPOSERS = 5 # select top K composers by SORT_BY type to train/test on
TO_SKIP = [] # composers to skip
# TO_SKIP = ['Frédéric Chopin', 'Franz Schubert', 'Ludwig van Beethoven', 'Johann Sebastian Bach', 'Franz Liszt']
AUGMENT_DATA = False

# tokenizer

USE_PRETRAINED_TOKENIZER = False
TOKENIZER_LOAD_PATH = Path("tokenizer.json").resolve() # pretrained tokenizer path

TRAIN_TOKENIZER = False # whether to train the tokenizer to a target vocab size with byte pair encoding
VOCAB_SIZE = 512 # target vocab size for tokenizer training
TOKENIZER_SAVE_PATH = Path("tokenizer.json").resolve()

BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack
    "num_tempos": 32,
    "tempo_range": (50, 200),  # (min_tempo, max_tempo)
}

# model

DIM = 144
DEPTH = 1
DIM_HEAD = 2
HEADS = 2
QK_SCALE = 2
FF_MULT = 2

ATTN_WINDOW_SIZES = [32, 64]

CONV_EXPANSION_FACTOR = 2
CONV_KERNEL_SIZE = 4

ATTN_DROPOUT = 0.1
FF_DROPOUT = 0.1
CONV_DROPOUT = 0.1

PRENORM = True

# pooling strategy for sequence-level classification
POOLING_STRATEGY = "sequence_attention"  # must be in ["sequence_attention", "mean", "max", "first"]

# training

NUM_EPOCHS = 20
BATCH_SIZE = 8

LEARNING_RATE = 4e-4
LR_SCHEDULER = None # must be in ["CosineAnnealingLR", "MultiStepLR", None]
MILESTONES = [12, 15] # for MultiStepLR
WEIGHT_DECAY = 4e-4 # 4e-4

MAX_GRAD_NORM = None # for gradient clipping, set to None to disable

# focal loss (set to False to use CE loss)

USE_FOCAL_LOSS = False
FOCAL_ALPHA = 2.0  # weighting factor for rare class
FOCAL_GAMMA = 3.0  # focusing parameter

# tensorboard log directory
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("logs") / f"k={TOP_K_COMPOSERS}_{datetime_str}"