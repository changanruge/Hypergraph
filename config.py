import torch

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = r'D:/程序实验/data/mat'
LABEL_FILE = r'D:/程序实验/label/Phenotypic_V1_0b_preprocessed1.csv'
MODEL_SAVE_DIR = r'D:/程序实验/model/hypergraph_new'

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.02
NUM_EPOCHS = 200
PATIENCE = 300

# Model Parameters
INDIM = 116  # Input dimension (number of ROIs)
NCLASS = 2   # Number of classes
D_MODEL = 32 # Hidden dimension
DROPOUT = 0.5

# Loss weights
LAMBDA_0 = 1    # Classification loss
LAMBDA_1 = 0    # Pool1 weight regularization
LAMBDA_2 = 0    # Pool2 weight regularization
LAMBDA_3 = 0.1  # TopK loss 1
LAMBDA_4 = 0.1  # TopK loss 2
LAMBDA_5 = 0.1  # Consistency loss

# Cross Validation
N_FOLDS = 10
RANDOM_SEED = 42

# Other
EPS = 1e-10
RATIO = 0.5
