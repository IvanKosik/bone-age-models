from pathlib import Path

from bsmu.bone_age.models import constants

IMAGE_DIR = Path('C:/MyDiskBackup/Projects/BoneAge/Data/SmallImages500_NoPads')

PART_DATA_CSV_PATH = Path(__file__).parent / 'data/part'
TRAIN_DATA_CSV_PATH = PART_DATA_CSV_PATH / 'train_with_predictions4.csv'
VALID_DATA_CSV_PATH = PART_DATA_CSV_PATH / 'valid_with_predictions4.csv'
### TEST_DATA_CSV_PATH = constants.TEST_DATA_CSV_PATH

BATCH_SIZE = 9
MODEL_NAME_PREFIX = 'CombinedSimple_DenseNet169_Xception_4Models'
MODEL_NAME_POSTFIX = 'Test500_NormalizedAge_Male_Test3'
