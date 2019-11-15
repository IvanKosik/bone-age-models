from pathlib import Path

from bsmu.bone_age.models import constants

IMAGE_DIR = Path('C:/MyDiskBackup/Projects/BoneAge/Data/SmallImages500_NoPads')

TRAIN_DATA_CSV_PATH = constants.PART_TRAIN_DATA_CSV_PATH
VALID_DATA_CSV_PATH = constants.PART_VALID_DATA_CSV_PATH
TEST_DATA_CSV_PATH = constants.TEST_DATA_CSV_PATH

BATCH_SIZE = 9
MODEL_NAME_PREFIX = 'Xception'
MODEL_NAME_POSTFIX = 'Test500_NormalizedAge'
