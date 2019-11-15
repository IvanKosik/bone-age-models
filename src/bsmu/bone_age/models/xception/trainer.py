import albumentations
import cv2
import keras
from keras import layers
from keras.applications import xception

from bsmu.bone_age.models import trainer
from bsmu.bone_age.models.xception import configs


class XceptionModelTrainer(trainer.ModelTrainer):
    IMAGE_DIR = configs.IMAGE_DIR
    TRAIN_DATA_CSV_PATH = configs.TRAIN_DATA_CSV_PATH
    VALID_DATA_CSV_PATH = configs.VALID_DATA_CSV_PATH
    TEST_DATA_CSV_PATH = configs.TEST_DATA_CSV_PATH

    BATCH_SIZE = configs.BATCH_SIZE
    MODEL_INPUT_IMAGE_SHAPE = (500, 500, 3)
    MODEL_NAME_PREFIX = configs.MODEL_NAME_PREFIX
    MODEL_NAME_POSTFIX = configs.MODEL_NAME_POSTFIX

    AUGMENTATION_TRANSFORMS = albumentations.Compose([
        albumentations.ShiftScaleRotate(
            border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.2, scale_limit=0.2,
            p=1.0),  # TODO try: interpolation=cv2.INTER_CUBIC
        albumentations.HorizontalFlip(p=0.5),
    ], p=1.0)

    INPUT_IMAGE_LAYER_NAME = 'input_1'
    INPUT_MALE_LAYER_NAME = 'input_male'
    OUTPUT_AGE_LAYER_NAME = 'output_age'
    OUTPUT_CONV_LAYER_NAME = 'block14_sepconv2_act'
    OUTPUT_POOLING_LAYER_NAME = 'encoder_pooling'

    def __init__(self, epochs: int = 100, lr: float = 1e-4):
        super().__init__(epochs, lr, preprocess_batch_images=xception.preprocess_input,
                         apply_age_normalization=True)

    def create_model(self):
        super().create_model()

        encoder_model = xception.Xception(include_top=False, weights='imagenet', pooling=None)
        input_image = encoder_model.input

        input_male = layers.Input(shape=(1,), name='input_male')

        x_image = encoder_model.output
        x_image = layers.GlobalAveragePooling2D(name='encoder_pooling')(x_image)

        x_male = layers.Dense(32, activation='relu')(input_male)

        x = layers.concatenate([x_image, x_male], axis=-1)
        x = layers.Dense(1000, activation='relu')(x)
        x = layers.Dense(1000, activation='relu')(x)
        output_age = layers.Dense(1, activation='linear', name='output_age')(x)

        self.model = keras.models.Model(inputs=[input_image, input_male], outputs=output_age)

    def _freeze_layers(self):
        super()._freeze_layers()

        self._unfreeze_all_layers()

        # for layer in self.model.layers[:-19]:
        #     layer.trainable = False

        self._print_layers_info()
