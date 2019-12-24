from pathlib import Path


IMAGE_DIR = Path('C:/MyDiskBackup/Projects/BoneAge/Data/SmallImages500_NoPads')

DATA_CSV_PATH = Path(__file__).parent / 'data'
ALL_DATA_CSV_PATH = DATA_CSV_PATH / 'all'
PART_DATA_CSV_PATH = DATA_CSV_PATH / 'part'

TRAIN_DATA_CSV_PATH = ALL_DATA_CSV_PATH / 'train_with_predictions2.csv'
VALID_DATA_CSV_PATH = ALL_DATA_CSV_PATH / 'valid_with_predictions2.csv'
TEST_DATA_CSV_PATH = ALL_DATA_CSV_PATH / 'test_with_predictions2.csv'

BATCH_SIZE = 9
MODEL_NAME_PREFIX = 'CombinedSimple_DenseNet169_Xception'
MODEL_NAME_POSTFIX = 'AllImages'
