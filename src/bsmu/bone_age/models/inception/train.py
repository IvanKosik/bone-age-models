import math
from pathlib import Path

import albumentations
import cv2
import keras
import numpy as np
import pandas as pd
import skimage.io
from keras import backend, layers, losses, metrics, optimizers
from keras.applications import inception_v3

from bsmu.bone_age.models import constants, debug_utils

BATCH_SIZE = 16

MODEL_INPUT_SIZE = (500, 500)
MODEL_INPUT_CHANNELS = 3
MODEL_NAME = f'Inception_{MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}_b{BATCH_SIZE}___Freeze2.h5'
MODEL_PATH = constants.MODEL_DIR / MODEL_NAME

LOG_PATH = constants.LOG_DIR / MODEL_NAME

IMAGE_DIR = Path('C:/MyDiskBackup/Projects/BoneAge/Data/SmallImages500_NoPads') # constants.IMAGE_DIR
TRAIN_DATA_CSV_PATH = constants.TRAIN_DATA_CSV_PATH
VALID_DATA_CSV_PATH = constants.VALID_DATA_CSV_PATH

AUGMENTATION = albumentations.Compose([
    albumentations.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.2, scale_limit=0.2,
                                    p=1.0),  # TODO try: interpolation=cv2.INTER_CUBIC
    albumentations.HorizontalFlip(p=0.5),
    ], p=1.0)


def normalized_image(image):
    image_min = image.min()
    if image_min != 0:
        image = image - image_min
    image_max = image.max()
    if image_max != 0 and image_max != 1:
        image = image / image_max
    return image


def augmentate_image(image):
    augmentation_results = AUGMENTATION(image=image)
    augmented_image = augmentation_results['image']

    # Normalize once again image to [0, 1]
    augmented_image = normalized_image(augmented_image)

    return augmented_image


class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir: Path, data_csv_path: Path, batch_size, is_train):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.is_train = is_train

        data_csv = pd.read_csv(str(data_csv_path))
        self.number_of_samples = len(data_csv.index)

        self.image_ids = np.zeros(shape=(self.number_of_samples, 1), dtype=np.uint32)
        self.males = np.zeros(shape=(self.number_of_samples, 1), dtype=np.uint8)
        self.ages = np.zeros(shape=(self.number_of_samples, 1), dtype=np.float32)

        for index, csv_row in enumerate(data_csv.values):
            image_id, age, male = csv_row
            print(f'#{index} image_id: {image_id} age: {age} male: {male}')

            self.image_ids[index, ...] = image_id
            self.males[index, ...] = male
            self.ages[index, ...] = age

        self.sample_indexes = np.arange(self.number_of_samples)
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch"""
        return math.ceil(self.number_of_samples / self.batch_size)

    def __getitem__(self, batch_index):
        """Generate one batch of data"""
        batch_images = np.zeros(shape=(self.batch_size, *MODEL_INPUT_SIZE, MODEL_INPUT_CHANNELS), dtype=np.float32)
        batch_males = np.zeros(shape=(self.batch_size, 1), dtype=np.uint8)
        batch_ages = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)

        # Generate image indexes of the batch
        batch_sample_indexes = self.sample_indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        for item_number, batch_sample_index in enumerate(batch_sample_indexes):
            image_id = self.image_ids[batch_sample_index][0]
            male = self.males[batch_sample_index][0]
            age = self.ages[batch_sample_index][0]

            image_path = self.image_dir / f'{image_id}.png'
            image = skimage.io.imread(str(image_path))
            image = normalized_image(image)

            if self.is_train:
                image = augmentate_image(image)

            image = image * 255
            image = np.stack((image,) * MODEL_INPUT_CHANNELS, axis=-1)
            batch_images[item_number, ...] = image

            batch_ages[item_number, ...] = age
            batch_males[item_number, ...] = male

        batch_images = inception_v3.preprocess_input(batch_images)
        return [batch_images, batch_males], batch_ages

    def on_epoch_end(self):
        """Shuffle files after each epoch"""
        if self.is_train:
            np.random.shuffle(self.sample_indexes)


'''
def age_mae(y_true, y_pred):
    y_true = (y_true + 1) * 120
    y_pred = (y_pred + 1) * 120
    return losses.mean_absolute_error(y_true, y_pred)
'''


def train_model(model, epochs=100, lr=1e-4):
    debug_utils.print_title(train_model.__name__)

    train_generator = DataGenerator(IMAGE_DIR, TRAIN_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=True)
    valid_generator = DataGenerator(IMAGE_DIR, VALID_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=False)

    monitor = 'val_loss'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str(MODEL_PATH), monitor=monitor,
                                                          verbose=1, save_best_only=True)
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.75, patience=3,
                                                           verbose=1, min_lr=1e-6)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor=monitor, patience=20)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(LOG_PATH), write_graph=False)

    callbacks = [checkpoint_callback, reduce_lr_callback, early_stopping_callback, tensorboard_callback]

    model.compile(optimizer=optimizers.Adam(lr=lr), loss=losses.mae, metrics=[metrics.mae])

    model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator)


def test_generator():
    debug_utils.print_title(test_generator.__name__)

    generator = DataGenerator(IMAGE_DIR, TRAIN_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=True)
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

'''
class StnResultsCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print('StnResultsCallback on_train_begin')

        self.generator = DataGenerator(IMAGES_PATH, TEST_STN_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=False)
        self.batch = self.generator.__getitem__(0)
        self.batch_input, self.batch_ages = self.batch
        self.batch_images, self.batch_males = self.batch_input[0], self.batch_input[1]

        self.batch1 = self.generator.__getitem__(1)
        self.batch_input1, self.batch_ages1 = self.batch1
        self.batch_images1, self.batch_males1 = self.batch_input1[0], self.batch_input1[1]

        self.number_of_images_to_save = 4  # len(self.batch_images)

        self.save_batch_images(self.batch_images, 0)
        self.save_batch_images(self.batch_images1, 1)

        input_1 = self.model.get_layer('input_1').input
        input_male = self.model.get_layer('input_male').input
        output_age = self.model.layers[-1].output

        # input_image = model.input
        output_STN = self.model.get_layer('stn_interpolation').output
        # STN_function = K.function([input_image], [output_STN])

        output_final_conv_layer = self.model.get_layer('xception').get_output_at(-1)  # Take last node from xception
        output_pooling_after_final_conv_layer = self.model.get_layer('encoder_pooling').output

        output_locnet_scale = self.model.get_layer('locnet_scale').output
        output_locnet_scale_activation = self.model.get_layer('locnet_scale_activation').output

        output_locnet_translate = self.model.get_layer('locnet_translate').output
        output_locnet_translate_activation = self.model.get_layer('locnet_translate_activation').output

        output_locnet_last_conv = self.model.get_layer('locnet_encoder').get_output_at(-1)
        output_locnet_pooling = self.model.get_layer('locnet_pooling').output

        self.STN_function = K.function(
            [input_1, input_male],
            [output_age, output_STN, output_final_conv_layer, output_pooling_after_final_conv_layer,
             output_locnet_scale, output_locnet_scale_activation, output_locnet_translate, output_locnet_translate_activation,
             output_locnet_last_conv, output_locnet_pooling])

    def save_batch_images(self, batch_images, batch_number):
        for index in range(self.number_of_images_to_save):
            image = batch_images[index]
            image = normalized_image(image)
            skimage.io.imsave(f'../Temp/STN_output/STN_b{batch_number}_{index}_input_norm.png', (image * 255).astype(np.int))

    def on_epoch_end(self, epoch, logs=None):
        print('\n\tStnResultsCallback on_epoch_end', epoch)

        self.stn_results_for_batch(self.batch_input, epoch, 0)
        self.stn_results_for_batch(self.batch_input1, epoch, 1)

    def build_cam(self, conv, pooling):
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

    def stn_results_for_batch(self, batch_input, epoch, batch_number):
        print(f'\n\t=============Batch {batch_number} epoch {epoch}')
        [age_results, stn_results, final_conv_results, pooling_after_final_conv_results,
         locnet_scale_results, locnet_scale_activations_results,
         locnet_translate_results, locnet_translate_activations_results,
         locnet_last_conv_results, locnet_pooling_results] = self.STN_function(batch_input)
        print_info(age_results, 'age_results info')
        print_info(stn_results, 'stn_results info')
        print_info(final_conv_results, 'final_conv_results info')
        print_info(pooling_after_final_conv_results, 'pooling_after_final_conv_results info')
        print_info(locnet_scale_results, 'locnet_scale_results')
        print_info(locnet_scale_activations_results, 'locnet_scale_activations_results')
        print_info(locnet_translate_results, 'locnet_translate_results')
        print_info(locnet_translate_activations_results, 'locnet_translate_activations_results')
        print_info(locnet_last_conv_results, 'locnet_last_conv_results')
        print_info(locnet_pooling_results, 'locnet_pooling_results')

        for index in range(self.number_of_images_to_save):
            input_image = batch_input[0][index, ...]

            print(f'\t---{index}---')
            age_result = (age_results[index, ...] + 1) * 120
            print('AGE', age_result)
            stn_result = stn_results[index, ...]
            print('stn result', stn_result.shape, stn_result.min(), stn_result.max(), stn_result.dtype)
            # skimage.io.imsave(f'../Temp/STN_output/STN_result{index}.png', stn_result)
            print('locnet_scale', locnet_scale_results[index, ...])
            print('locnet_scale_activation', locnet_scale_activations_results[index, ...])
            print('locnet_translate', locnet_translate_results[index, ...])
            print('locnet_translate_activation', locnet_translate_activations_results[index, ...])

            normalized_stn_result = normalized_image(stn_result)
            stn_save_file_name = Path(f'STN_b{batch_number}_{index}_result_norm_epoch_{epoch}.png')
            ### skimage.io.imsave(str(STN_OUTPUT_PATH / stn_save_file_name), (normalized_stn_result * 255).astype(np.int))

            # Save activation map
            cam = self.build_cam(final_conv_results[index, ...], pooling_after_final_conv_results[index, ...])
            
            cam = cv2.resize(cam, SAMPLING_SIZE, interpolation=cv2.INTER_LINEAR)
            ### skimage.io.imsave(str(STN_OUTPUT_PATH / (stn_save_file_name.name + '_cam.png')), cam)
            # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap[np.where(cam < 0.05)] = 0
            img = heatmap * 0.5 + (normalized_stn_result * 255)
            cv2.imwrite(str(STN_OUTPUT_PATH / (stn_save_file_name.name + '_cam_heat.png')), img)

            # Save locnet activation map
            locnet_cam = self.build_cam(locnet_last_conv_results[index, ...], locnet_pooling_results[index, ...])

            locnet_cam = cv2.resize(locnet_cam, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            ### skimage.io.imsave(str(STN_OUTPUT_PATH / (stn_save_file_name.name + 'locnet_cam.png')), locnet_cam)
            locnet_heatmap = cv2.applyColorMap((255 * locnet_cam).astype(np.uint8), cv2.COLORMAP_JET)
            locnet_heatmap[np.where(locnet_cam < 0.05)] = 0
            locnet_img = locnet_heatmap * 0.5 + (normalized_image(input_image) * 255)
            cv2.imwrite(str(STN_OUTPUT_PATH / (stn_save_file_name.name + '_locnet_cam_heat.png')), locnet_img)
'''


def create_model():
    debug_utils.print_title(create_model.__name__)

    # input_image = layers.Input(shape=(*MODEL_INPUT_SIZE, MODEL_INPUT_CHANNELS), name='input_image')
    input_male = layers.Input(shape=(1,), name='input_male')

    encoder_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', pooling=None)
    # encoder_model.name = 'encoder_model'
    input_image = encoder_model.input

    #x_image = encoder_model(input_image)
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

    # test_generator()
    # exit()

    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
    else:
        model = create_model()





    # Set all layers to trainable
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
    # for layer in model.layers[:-25]:
    #     layer.trainable = False

    print('================== After FREZE ===============')
    for i in range(len(model.layers)):
        print(i, model.layers[i], '      --- ', model.layers[i].name, model.layers[i].trainable)







    debug_utils.print_title('model summary')
    model.summary()

    train_model(model)


if __name__ == '__main__':
    main()
