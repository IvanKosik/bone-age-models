import math
import typing
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import skimage.io
from keras import backend, optimizers, losses, metrics

from bsmu.bone_age.models import constants
from bsmu.bone_age.models import debug_utils


class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir: Path, data_csv_path: Path, batch_size: int, input_image_shape: tuple,
                 shuffle: bool, preprocess_batch_images: typing.Callable, augmentation_transforms=None):
        assert len(input_image_shape) == 3, 'Input image shape must have 3 dims: width, height and number of channels'

        self.image_dir = image_dir
        self.batch_size = batch_size
        self.input_image_shape = input_image_shape
        self.shuffle = shuffle
        self.preprocess_batch_images = preprocess_batch_images
        self.augmentation_transforms = augmentation_transforms

        data_csv = pd.read_csv(str(data_csv_path))
        self.number_of_samples = len(data_csv.index)

        self.image_ids = np.zeros(shape=(self.number_of_samples, 1), dtype=np.uint32)
        self.males = np.zeros(shape=(self.number_of_samples, 1), dtype=np.uint8)
        self.ages = np.zeros(shape=(self.number_of_samples, 1), dtype=np.float32)

        for index, csv_row in enumerate(data_csv.values):
            image_id, male, age = csv_row
            print(f'#{index} image_id: {image_id} male: {male} age: {age}')

            self.image_ids[index, 0] = image_id
            self.males[index, 0] = male
            self.ages[index, 0] = age

        self.sample_indexes = np.arange(self.number_of_samples)
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch"""
        return math.ceil(self.number_of_samples / self.batch_size)

    def __getitem__(self, batch_index):
        """Generate one batch of data"""
        batch_images = np.zeros(shape=(self.batch_size, *self.input_image_shape), dtype=np.float32)
        batch_males = np.zeros(shape=(self.batch_size, 1), dtype=np.uint8)
        batch_ages = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)

        # Generate image indexes of the batch
        batch_sample_indexes = self.sample_indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        for item_number, batch_sample_index in enumerate(batch_sample_indexes):
            image_id = self.image_ids[batch_sample_index, 0]
            male = self.males[batch_sample_index, 0]
            age = self.ages[batch_sample_index, 0]

            image_path = self.image_dir / f'{image_id}.png'
            image = skimage.io.imread(str(image_path))
            image = normalized_image(image)

            if self.augmentation_transforms is not None:
                image = augmentate_image(image, self.augmentation_transforms)

                # Normalize once again image to [0, 1] after augmentation
                image = normalized_image(image)

            image = image * 255
            image = np.stack((image,) * self.input_image_shape[2], axis=-1)
            batch_images[item_number, ...] = image

            batch_ages[item_number, 0] = age
            batch_males[item_number, 0] = male

        batch_images = self.preprocess_batch_images(batch_images)
        return [batch_images, batch_males], batch_ages

    def on_epoch_end(self):
        """Shuffle files after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.sample_indexes)


def normalized_image(image):
    image_min = image.min()
    if image_min != 0:
        image = image - image_min
    image_max = image.max()
    if image_max != 0 and image_max != 1:
        image = image / image_max
    return image


def augmentate_image(image, transforms):
    augmentation_results = transforms(image=image)
    return augmentation_results['image']


def train_model(model, train_generator, valid_generator, model_path, log_path, epochs=100, lr=1e-4):
    debug_utils.print_title(train_model.__name__)

    monitor = 'val_loss'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor=monitor,
                                                          verbose=1, save_best_only=True)
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.75, patience=3,
                                                           verbose=1, min_lr=1e-6)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor=monitor, patience=20)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(log_path), write_graph=False)

    callbacks = [checkpoint_callback, reduce_lr_callback, early_stopping_callback, tensorboard_callback]

    model.compile(optimizer=optimizers.Adam(lr=lr), loss=losses.mae, metrics=[metrics.mae])

    model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator)


def test_generator(generator):
    debug_utils.print_title(test_generator.__name__)

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

        image = normalized_image(image)

        skimage.io.imsave(str(constants.TEST_GENERATOR_DIR / f'{batch_image_index}.png'), image)


def model_output_function(model, input_layer_names: list, output_layer_names: list):
    inputs = [model.get_layer(input_layer_name).input for input_layer_name in input_layer_names]
    outputs = [model.get_layer(output_layer_name).output for output_layer_name in output_layer_names]
    return backend.function(inputs, outputs)


def build_cam(conv, pooling):
    # cam = np.zeros(shape=conv.shape[:2], dtype=np.float32)
    cam = np.copy(conv)
    for feature_map_index in range(cam.shape[2]):
        cam[..., feature_map_index] *= pooling[feature_map_index]

    # print_info(cam, 'cam ---0---')
    cam = np.mean(cam, axis=-1)
    # print_info(cam, 'cam ---1--- mean')
    cam = np.maximum(cam, 0)
    # print_info(cam, 'cam ---2--- maximum')
    cam = cam / np.max(cam)
    # print_info(cam, 'cam ---3--- devide to max')
    return cam
