import keras
import skimage.io
from keras import optimizers, losses, metrics

from bsmu.bone_age.models import constants, debug_utils, train_utils


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

    PREPROCESS_BATCH_IMAGES = None

    def __init__(self, epochs: int = 100, lr: float = 1e-4):
        self.epochs = epochs
        self.lr = lr

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

    def run(self):
        if self.model_path.exists():
            self._load_model()
        else:
            self._create_model()

        self.train_model()

    def _create_model(self):
        debug_utils.print_title(self._create_model.__name__)

    def _load_model(self):
        debug_utils.print_title(self._load_model.__name__)

        self.model = keras.models.load_model(str(self.model_path), compile=False)

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
            shuffle=True, preprocess_batch_images=self.PREPROCESS_BATCH_IMAGES,
            augmentation_transforms=self.AUGMENTATION_TRANSFORMS)

    @property
    def valid_generator(self):
        if self._valid_generator is None:
            self._create_valid_generator()
        return self._valid_generator

    def _create_valid_generator(self):
        self._valid_generator = train_utils.DataGenerator(
            self.IMAGE_DIR, self.VALID_DATA_CSV_PATH, self.BATCH_SIZE, self.MODEL_INPUT_IMAGE_SHAPE,
            shuffle=False, preprocess_batch_images=self.PREPROCESS_BATCH_IMAGES)

    @property
    def test_generator(self):
        if self._test_generator is None:
            self._create_test_generator()
        return self._test_generator

    def _create_test_generator(self):
        self._test_generator = train_utils.DataGenerator(
            self.IMAGE_DIR, self.TEST_DATA_CSV_PATH, self.BATCH_SIZE, self.MODEL_INPUT_IMAGE_SHAPE,
            shuffle=False, preprocess_batch_images=self.PREPROCESS_BATCH_IMAGES)

    def train_model(self):
        debug_utils.print_title(self.train_model.__name__)

        self.model.compile(optimizer=optimizers.Adam(lr=self.lr), loss=losses.mae, metrics=[metrics.mae])
        self.model.fit_generator(generator=self.train_generator,
                                 epochs=self.epochs,
                                 callbacks=self.callbacks,
                                 validation_data=self.valid_generator)

    def verify_generator(self, generator):
        debug_utils.print_title(self.test_generator.__name__)

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

            image = train_utils.normalized_image(image)

            skimage.io.imsave(str(constants.TEST_GENERATOR_DIR / f'{batch_image_index}.png'), image)

    def generate_activation_map(self):
        ...

    def test_model(self):
        ...
