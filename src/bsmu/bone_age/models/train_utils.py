import math
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import skimage.io


class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir: Path, data_csv_path: Path, batch_size, input_image_shape,
                 shuffle, preprocess_batch_images, augmentation_transforms=None):
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
            image_id, age, male = csv_row
            print(f'#{index} image_id: {image_id} age: {age} male: {male}')

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
