from pathlib import Path
from datetime import datetime

SEED = 42
DEVICE = (
    'cuda' if __import__('torch').cuda.is_available() else
    'mps' if __import__('torch').backends.mps.is_available() else
    'cpu'
)

# data paths

MAESTRO_DATA_PATH = Path("data/maestro-v3.0.0").resolve()
MAESTRO_CSV = MAESTRO_DATA_PATH / "maestro-v3.0.0.csv"

# data splitting

SPLIT_DATA = True
SHUFFLE = True
SORT_BY = 'compositions' # must be in ['compositions', 'duration']
TEST_SIZE = 0.1
TOP_K_COMPOSERS = 3 # select top k composers by SORT_BY type to train/test on
MIN_COMPOSER_DURATION = 10000 # unused
TO_SKIP = [] # composers to skip
AUGMENT_DATA = False

# tokenizer

TOKENIZER_LOAD_PATH = Path("tokenizer.json").resolve() # pretrained tokenizer path

TRAIN_TOKENIZER = True # whether to train a new tokenizer
TOKENIZER_SAVE_PATH = Path("tokenizer.json").resolve()

VOCAB_SIZE = 30000
BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 8,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack
    "num_tempos": 16,
    "tempo_range": (50, 200),  # (min_tempo, max_tempo)
}

# training

NUM_EPOCHS = 20
BATCH_SIZE = 8

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-4

MAX_SEQ_LEN = 1024

LOG_DIR = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S") # tensorboard log directory