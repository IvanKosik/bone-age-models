import typing

import keras
import numpy as np
import skimage.io
from keras import optimizers, losses, metrics

from bsmu.bone_age.models import constants, debug_utils, train_utils, cam_utils, image_utils


class ModelTrainer:
    IMAGE_DIR = constants.IMAGE_DIR
    TRAIN_DATA_CSV_PATH = constants.TRAIN_DATA_CSV_PATH
    VALID_DATA_CSV_PATH = constants.VALID_DATA_CSV_PATH
    TEST_DATA_CSV_PATH = constants.TEST_DATA_CSV_PATH

    BATCH_SIZE = 16
    MODEL_INPUT_IMAGE_SHAPE = (500, 500, 3)
    MODEL_NAME_PREFIX = ''
    MODEL_NAME_POSTFIX = ''

    AUGMENTATION_TRANSFORMS = None

    INPUT_IMAGE_LAYER_NAME = ''
    INPUT_MALE_LAYER_NAME = ''
    OUTPUT_AGE_LAYER_NAME = ''
    OUTPUT_CONV_LAYER_NAME = ''
    OUTPUT_POOLING_LAYER_NAME = ''

    def __init__(self, epochs: int = 100, lr: float = 1e-4, preprocess_batch_images: typing.Callable = None):
        self.epochs = epochs
        self.lr = lr
        self.preprocess_batch_images = preprocess_batch_images

        self.model = None

        self._train_generator = None
        self._valid_generator = None
        self._test_generator = None

        self.callbacks = self._create_callbacks()

    @property
    def model_name(self):
        size = f'{self.MODEL_INPUT_IMAGE_SHAPE[0]}x{self.MODEL_INPUT_IMAGE_SHAPE[1]}'
        batch_size = f'b{self.BATCH_SIZE}'
        name_parts = [self.MODEL_NAME_PREFIX, size, batch_size, self.MODEL_NAME_POSTFIX]
        name = '_'.join(filter(None, name_parts))
        return f'{name}.h5'

    @property
    def model_path(self):
        return constants.MODEL_DIR / self.model_name

    @property
    def log_path(self):
        return constants.LOG_DIR / self.model_name

    @property
    def model_input_image_size(self):
        return self.MODEL_INPUT_IMAGE_SHAPE[:2]

    @property
    def model_input_image_channels_count(self):
        return self.MODEL_INPUT_IMAGE_SHAPE[2]

    def run(self):
        if self.model_path.exists():
            self.load_model()
        else:
            self.create_model()

        self._freeze_layers()

        self._train_model()

    def create_model(self):
        debug_utils.print_title(self.create_model.__name__)

    def load_model(self):
        debug_utils.print_title(self.load_model.__name__)

        self.model = keras.models.load_model(str(self.model_path), compile=False)

    def _freeze_layers(self):
        debug_utils.print_title(self._freeze_layers.__name__)

    def _unfreeze_all_layers(self):
        for layer in self.model.layers:
            layer.trainable = True

    def _create_callbacks(self):
        monitor = 'val_loss'
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str(self.model_path), monitor=monitor,
                                                              verbose=1, save_best_only=True)
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.75, patience=3,
                                                               verbose=1, min_lr=1e-6)
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor=monitor, patience=20)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(self.log_path), write_graph=False)

        return [checkpoint_callback, reduce_lr_callback, early_stopping_callback, tensorboard_callback]

    @property
    def train_generator(self):
        if self._train_generator is None:
            self._create_train_generator()
        return self._train_generator

    def _create_train_generator(self):
        self._train_generator = train_utils.DataGenerator(
            self.IMAGE_DIR, self.TRAIN_DATA_CSV_PATH, self.BATCH_SIZE, self.MODEL_INPUT_IMAGE_SHAPE,
            shuffle=True, preprocess_batch_images=self.preprocess_batch_images,
            augmentation_transforms=self.AUGMENTATION_TRANSFORMS)

    @property
    def valid_generator(self):
        if self._valid_generator is None:
            self._create_valid_generator()
        return self._valid_generator

    def _create_valid_generator(self):
        self._valid_generator = train_utils.DataGenerator(
            self.IMAGE_DIR, self.VALID_DATA_CSV_PATH, self.BATCH_SIZE, self.MODEL_INPUT_IMAGE_SHAPE,
            shuffle=False, preprocess_batch_images=self.preprocess_batch_images)

    @property
    def test_generator(self):
        if self._test_generator is None:
            self._create_test_generator()
        return self._test_generator

    def _create_test_generator(self):
        self._test_generator = train_utils.DataGenerator(
            self.IMAGE_DIR, self.TEST_DATA_CSV_PATH, self.BATCH_SIZE, self.MODEL_INPUT_IMAGE_SHAPE,
            shuffle=False, preprocess_batch_images=self.preprocess_batch_images)

    def _train_model(self):
        debug_utils.print_title(self._train_model.__name__)

        self.model.compile(optimizer=optimizers.Adam(lr=self.lr), loss=losses.mae, metrics=[metrics.mae])
        self.model.fit_generator(generator=self.train_generator,
                                 epochs=self.epochs,
                                 callbacks=self.callbacks,
                                 validation_data=self.valid_generator)

    def _print_layers_info(self):
        debug_utils.print_title(self._print_layers_info.__name__)
        debug_utils.print_layers_info(self.model)

    def verify_generator(self, generator):
        debug_utils.print_title(self.verify_generator.__name__)

        generator_len = len(generator)
        print('generator_len (number of batches per epoch):', generator_len)

        batch = generator.__getitem__(int(generator_len / 2))
        batch_input, batch_ages = batch
        batch_images, batch_males = batch_input[0], batch_input[1]
        debug_utils.print_info(batch_images, 'batch_images')
        debug_utils.print_info(batch_males, 'batch_males')
        debug_utils.print_info(batch_ages, 'batch_ages')

        # Save all images in batches
        for batch_image_index in range(len(batch_images)):
            image = batch_images[batch_image_index][...]
            male = batch_males[batch_image_index][0]
            age = batch_ages[batch_image_index][0]

            debug_utils.print_info(image, 'image')
            print(male, 'male')
            print(age, 'age')

            image = image_utils.normalized_image(image)

            skimage.io.imsave(str(constants.TEST_GENERATOR_DIR / f'{batch_image_index}.png'), image)

    def create_batch_from_one_sample(self, image, male: bool):
        assert len(image.shape) == 2, 'one channel images are only supported'
        assert image.shape == self.model_input_image_size, 'image size is not equal to model input size'

        # Create a batch from one image
        batch_images = np.zeros(shape=(self.BATCH_SIZE, *self.MODEL_INPUT_IMAGE_SHAPE), dtype=np.float32)
        batch_males = np.zeros(shape=(self.BATCH_SIZE, 1), dtype=np.uint8)

        image = image_utils.normalized_image(image)
        image = image * 255
        image = np.stack((image,) * self.model_input_image_channels_count, axis=-1)
        batch_images[0, ...] = image

        batch_males[0, 0] = male

        batch_images = self.preprocess_batch_images(batch_images)

        input_batch = [batch_images, batch_males]
        return input_batch

    def generate_image_cam(self, image, male: bool):
        input_batch = self.create_batch_from_one_sample(image, male)
        cam_batch, output_age_batch = self.generate_cam_batch(input_batch)
        return cam_batch[0], output_age_batch[0]

    def generate_cam_batch(self, input_batch):
        assert self.INPUT_IMAGE_LAYER_NAME and self.INPUT_MALE_LAYER_NAME \
               and self.OUTPUT_AGE_LAYER_NAME and self.OUTPUT_CONV_LAYER_NAME and self.OUTPUT_POOLING_LAYER_NAME, \
               'define all needed layer names to generate activation map'
        return cam_utils.generate_cam_batch(
            input_batch, self.model, self.INPUT_IMAGE_LAYER_NAME, self.INPUT_MALE_LAYER_NAME,
            self.OUTPUT_AGE_LAYER_NAME, self.OUTPUT_CONV_LAYER_NAME, self.OUTPUT_POOLING_LAYER_NAME)

    def generate_image_cam_overlay(self, image, male: bool):
        cam, age = self.generate_image_cam(image, male)
        overlay_result = cam_utils.overlay_cam(image, cam)
        return overlay_result, age

    def test_model(self):
        ...
