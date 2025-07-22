from miditok import REMI, TokenizerConfig
from pathlib import Path

# tokenizer config

from config import TOKENIZER_SAVE_PATH, VOCAB_SIZE

midi_paths = list(Path("data/maestro-v3.0.0").resolve().glob("*/*.mid?"))

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

# create tokenizer

config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

# train the tokenizer with Byte Pair Encoding (BPE) to build the vocabulary

tokenizer.train(
    vocab_size=VOCAB_SIZE,
    files_paths=midi_paths,
)
tokenizer.save(TOKENIZER_SAVE_PATH)