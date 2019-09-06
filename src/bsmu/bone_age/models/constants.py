from pathlib import Path

PROJECT_DIR = Path(__file__).parents[4].resolve()
DATA_DIR = PROJECT_DIR / 'data'
IMAGE_DIR = DATA_DIR / 'images'
OUTPUT_DIR = PROJECT_DIR / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'
LOG_DIR = OUTPUT_DIR / 'logs'
TEST_GENERATOR_DIR = OUTPUT_DIR / 'test_generator'

CSV_DIR = DATA_DIR / 'csv'
ALL_IMAGES_CSV_DIR = CSV_DIR / 'all'
TRAIN_DATA_CSV_PATH = ALL_IMAGES_CSV_DIR / 'train.csv'
VALID_DATA_CSV_PATH = ALL_IMAGES_CSV_DIR / 'valid.csv'
TEST_DATA_CSV_PATH = ALL_IMAGES_CSV_DIR / 'test.csv'
