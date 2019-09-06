from pathlib import Path

PROJECT_DIR = Path(__file__).parents[4].resolve()
DATA_DIR = PROJECT_DIR / 'data'
IMAGE_DIR = DATA_DIR / 'images'
OUTPUT_DIR = PROJECT_DIR / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'
LOG_DIR = OUTPUT_DIR / 'logs'
TEST_GENERATOR_DIR = OUTPUT_DIR / 'test_generator'

TRAIN_DATA_CSV_PATH = DATA_DIR / 'train.csv'
VALID_DATA_CSV_PATH = DATA_DIR / 'valid.csv'
TEST_DATA_CSV_PATH = DATA_DIR / 'test.csv'
