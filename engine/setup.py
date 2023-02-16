# Dataset
DATASET_DIR = "dataset/"
N_ROWS = 250_000

### Regular positions from games on lichess
DATASET_REGULAR = "chessData.csv" 
DATASET_REGULAR_SIZE = 0.33

### Positions reached by making a random move in a regular position
DATASET_RANDOM = "random_evals.csv" 
DATASET_RANDOM_SIZE = 0.33

### Heavily tactical positions where a combination of moves is required to win
DATASET_TACTIC = "tactic_evals.csv" 
DATASET_TACTIC_SIZE = 0.33

N_ROWS_EFFECTIVE = min(N_ROWS*DATASET_REGULAR_SIZE, 12_954_834) + \
                    min(N_ROWS*DATASET_RANDOM_SIZE, 1_000_273) + \
                    min(N_ROWS*DATASET_TACTIC_SIZE, 2_628_219)

DATASET_VECTORIZED = "dataset_vectorized.csv" 
TRAINING_SET_SIZE = 0.8
FORCED_MATE_CENTIPAWN = 10_000
CENTIPAWN_CLAMP = 10_000

# NN architecture
N_FEATURES = 8*8*12
HIDDEN_ACTIVATION = "elu"
OUTPUT_ACTIVATION = "relu"

# Training
LEARNING_RATE = 0.0001
REGULARIZATION_RATE = 0
EPOCHS = 40
BATCH_SIZE = 250
EARLY_STOPPING = False
PATIENCE = 3

# Performance evaluation
K_FOLD = False
N_FOLDS = 3

# Save/load models
SAVE_MODEL = True
MODEL_NAME = "models/latest"
