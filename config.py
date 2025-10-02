import os

# כללי
SEED = 42

# נתיב לדאטא
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# היפר-פרמטרים
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# פלט
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'model.pt')
LOSS_PLOT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'loss_plot.png')
