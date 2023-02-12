# Dataset
DATASET_DIR = "dataset/"
DATASET = "chessData.csv" 
DATASET_VECTORIZED = "chessData_vec.csv" 
N_ROWS = 300_000
TRAINING_SET_SIZE = 0.8
FORCED_MATE_CENTIPAWN = 10_000
CENTIPAWN_CLAMP = 1_000

# NN architecture
N_FEATURES = 8*8*12
HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "relu"

# Training
LEARNING_RATE = 0.0001
REGULARIZATION_RATE = 0
EPOCHS = 30
BATCH_SIZE = 32
EARLY_STOPPING = True
PATIENCE = 3

# Performance evaluation
K_FOLD = False
N_FOLDS = 3

# Save/load models
SAVE_MODEL = True
MODEL_NAME = "models/latest"
