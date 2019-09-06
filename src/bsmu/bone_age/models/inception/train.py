from pathlib import Path

import albumentations
import cv2
import keras
from keras import backend, layers
from keras.applications import inception_v3

from bsmu.bone_age.models import constants, debug_utils
from bsmu.bone_age.models import train_utils

BATCH_SIZE = 16

MODEL_INPUT_IMAGE_SHAPE = (500, 500, 3)
MODEL_NAME = f'Inception_{MODEL_INPUT_IMAGE_SHAPE[0]}x{MODEL_INPUT_IMAGE_SHAPE[1]}_b{BATCH_SIZE}___Freeze7_500_round2.h5'
MODEL_PATH = constants.MODEL_DIR / MODEL_NAME

LOG_PATH = constants.LOG_DIR / MODEL_NAME

IMAGE_DIR = Path('C:/MyDiskBackup/Projects/BoneAge/Data/SmallImages500_NoPads') # constants.IMAGE_DIR
TRAIN_DATA_CSV_PATH = constants.TRAIN_DATA_CSV_PATH
VALID_DATA_CSV_PATH = constants.VALID_DATA_CSV_PATH

AUGMENTATION_TRANSFORMS = albumentations.Compose([
    albumentations.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.2, scale_limit=0.2,
                                    p=1.0),  # TODO try: interpolation=cv2.INTER_CUBIC
    albumentations.HorizontalFlip(p=0.5),
    ], p=1.0)

PREPROCESS_BATCH_IMAGES = inception_v3.preprocess_input


def create_model():
    debug_utils.print_title(create_model.__name__)

    encoder_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', pooling=None)
    input_image = encoder_model.input

    input_male = layers.Input(shape=(1,), name='input_male')

    x_image = encoder_model.output
    x_image = layers.GlobalAveragePooling2D(name='encoder_pooling')(x_image)

    x_male = layers.Dense(32, activation='relu')(input_male)

    x = layers.concatenate([x_image, x_male], axis=-1)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(1000, activation='relu')(x)
    output_age = layers.Dense(1, activation='linear', name='output_age')(x)

    model = keras.models.Model(inputs=[input_image, input_male], outputs=output_age)
    return model


def load_model(path: Path):
    debug_utils.print_title(load_model.__name__)

    model = keras.models.load_model(str(path), compile=False)
    return model


def main():
    print(f'Tensorflow version: {backend.tf.__version__}')

    # train_utils.test_generator(train_utils.DataGenerator(
    #     IMAGE_DIR, TRAIN_DATA_CSV_PATH, BATCH_SIZE, MODEL_INPUT_IMAGE_SHAPE, shuffle=True,
    #     preprocess_batch_images=PREPROCESS_BATCH_IMAGES, augmentation_transforms=AUGMENTATION_TRANSFORMS))
    # exit()

    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
    else:
        model = create_model()



    # # Set all layers to trainable
    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    # encoder_pooling_layer = model.get_layer('encoder_pooling')
    # encoder_pooling_layer = model.layers.index(encoder_pooling_layer)
    #
    # # for i in range(encoder_pooling_layer + 1, len(model.layers)):
    # #     model.layers[i].trainable = False
    #
    # # for i in range(encoder_pooling_layer):
    # #     model.layers[i].trainable = False
    #
    # for layer in model.layers[:-7]:
    #     layer.trainable = False

    print('================== After FREZE ===============')
    for i in range(len(model.layers)):
        print(i, model.layers[i], '      --- ', model.layers[i].name, model.layers[i].trainable)





    debug_utils.print_title('model summary')
    model.summary()

    train_utils.train_model(
        model,
        train_generator=train_utils.DataGenerator(IMAGE_DIR, TRAIN_DATA_CSV_PATH, BATCH_SIZE, MODEL_INPUT_IMAGE_SHAPE,
                                                  shuffle=True, preprocess_batch_images=PREPROCESS_BATCH_IMAGES,
                                                  augmentation_transforms=AUGMENTATION_TRANSFORMS),
        valid_generator=train_utils.DataGenerator(IMAGE_DIR, VALID_DATA_CSV_PATH, BATCH_SIZE, MODEL_INPUT_IMAGE_SHAPE,
                                                  shuffle=False, preprocess_batch_images=PREPROCESS_BATCH_IMAGES),
        model_path=MODEL_PATH, log_path=LOG_PATH)


if __name__ == '__main__':
    main()
