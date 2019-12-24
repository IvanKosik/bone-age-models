from pathlib import Path

from bsmu.bone_age.models import constants

IMAGE_DIR = Path('C:/MyDiskBackup/Projects/BoneAge/Data/SmallImages500_NoPads')

TRAIN_DATA_CSV_PATH = constants.PART_TRAIN_DATA_CSV_PATH
VALID_DATA_CSV_PATH = constants.PART_VALID_DATA_CSV_PATH
TEST_DATA_CSV_PATH = constants.TEST_DATA_CSV_PATH

BATCH_SIZE = 7
MODEL_NAME_PREFIX = 'DenseNet169'
MODEL_NAME_POSTFIX = 'AllImages'
