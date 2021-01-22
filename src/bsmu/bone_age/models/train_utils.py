import math
import typing
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import skimage.io
from keras import losses

from bsmu.bone_age.models import image_utils


class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir: Path, data_csv_path: Path, batch_size: int, input_image_shape: tuple,
                 shuffle: bool, preprocess_batch_images: typing.Callable, augmentation_transforms=None,
                 apply_age_normalization: bool = True, combined_model: bool = False,
                 discard_last_incomplete_batch: bool = True):
        assert len(input_image_shape) == 3, 'Input image shape must have 3 dims: width, height and number of channels'

        self.image_dir = image_dir
        self.batch_size = batch_size
        self.input_image_shape = input_image_shape
        self.shuffle = shuffle
        self.preprocess_batch_images = preprocess_batch_images
        self.augmentation_transforms = augmentation_transforms
        self.combined_model = combined_model
        self.discard_last_incomplete_batch = discard_last_incomplete_batch

        data_frame = pd.read_csv(str(data_csv_path))
        data = data_frame.to_numpy()
        if not combined_model:
            data = data[:, :3]  # remove prediction columns
        self.number_of_samples = len(data)

        self.image_ids = np.zeros(shape=(self.number_of_samples, 1), dtype=object)
        self.males = np.zeros(shape=(self.number_of_samples, 1), dtype=np.uint8)
        self.ages = np.zeros(shape=(self.number_of_samples, 1), dtype=np.float32)
        self.number_of_predictions = data.shape[1] - 3  # CSV can contain predictions of other models in the last columns
        self.predictions = np.zeros(shape=(self.number_of_samples, self.number_of_predictions), dtype=np.float32)

        for index, data_row in enumerate(data):
            image_id, male, age, *self.predictions[index] = data_row
            print(f'#{index} image_id: {image_id} male: {male} age: {age} \t\tpredictions: {self.predictions[index]}')
            self.image_ids[index, 0] = image_id if '.' in str(image_id) else f'{image_id}.png'
            self.males[index, 0] = male
            self.ages[index, 0] = age

        self._zero_image_age = 0
        if apply_age_normalization:
            self.ages = normalized_age(self.ages)
            self.predictions = normalized_age(self.predictions)
            self._zero_image_age = normalized_age(self._zero_image_age)

        self.sample_indexes = np.arange(self.number_of_samples)
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch"""
        number_of_batches = self.number_of_samples / self.batch_size
        return math.floor(number_of_batches) if self.discard_last_incomplete_batch else math.ceil(number_of_batches)

    def __getitem__(self, batch_index):
        """Generate one batch of data"""
        batch_images = np.zeros(shape=(self.batch_size, *self.input_image_shape), dtype=np.float32)
        batch_males = np.zeros(shape=(self.batch_size, 1), dtype=np.uint8)
        batch_ages = np.full(shape=(self.batch_size, 1), fill_value=self._zero_image_age, dtype=np.float32)
        batch_predictions = np.full(shape=(self.batch_size, self.number_of_predictions),
                                    fill_value=self._zero_image_age, dtype=np.float32)

        # Generate image indexes of the batch
        batch_sample_indexes = self.sample_indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        for item_number, batch_sample_index in enumerate(batch_sample_indexes):
            image_id = self.image_ids[batch_sample_index, 0]
            male = self.males[batch_sample_index, 0]
            age = self.ages[batch_sample_index, 0]

            image_path = self.image_dir / image_id
            image = skimage.io.imread(str(image_path))
            image = image_utils.normalized_image(image).astype(np.float32)

            if self.augmentation_transforms is not None:
                image = augmentate_image(image, self.augmentation_transforms)

                # Normalize once again image to [0, 1] after augmentation
                image = image_utils.normalized_image(image)

            image = image * 255
            image = np.stack((image,) * self.input_image_shape[2], axis=-1)
            if not self.combined_model:
                batch_images[item_number, ...] = image

            batch_ages[item_number, 0] = age
            batch_males[item_number, 0] = male
            batch_predictions[item_number] = np.copy(self.predictions[batch_sample_index])

        if self.preprocess_batch_images is not None:
            batch_images = self.preprocess_batch_images(batch_images)

        batch_input = [batch_males, batch_predictions] if self.combined_model else [batch_images, batch_males]
        return batch_input, batch_ages

    def on_epoch_end(self):
        """Shuffle files after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.sample_indexes)


def augmentate_image(image, transforms):
    augmentation_results = transforms(image=image)
    return augmentation_results['image']


def normalized_age(age):
    """
    Normalize age to [-1, 1]
    """
    return age / 120 - 1


def denormalized_age(age):
    """
    :param age: age in range [-1, 1]
    :return: age in range [0, 240] (months)
    """
    return (age + 1) * 120


def age_mae(y_true, y_pred):
    y_true = denormalized_age(y_true)
    y_pred = denormalized_age(y_pred)
    return losses.mean_absolute_error(y_true, y_pred)
