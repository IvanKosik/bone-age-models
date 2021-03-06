from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform

from bsmu.bone_age.models import debug_utils
from bsmu.bone_age.models import image_utils
from bsmu.bone_age.models import latin_utils

# from bsmu.bone_age.models.combined.simple import trainer
from bsmu.bone_age.models.dense_net import trainer
# from bsmu.bone_age.models.inception import trainer
# from bsmu.bone_age.models.xception import trainer
# from bsmu.bone_age.models.stn import trainer
from bsmu.bone_age.models import constants


PROJECT_PATH = Path('../../../../')
TEMP_PATH = PROJECT_PATH / 'temp'
OUTPUT_PATH = PROJECT_PATH / 'output'


def test_image(model_trainer, image_path, output_path=OUTPUT_PATH):
    image_src = skimage.io.imread(image_path, as_gray=True)
    print(f'\n\t\t--------- {image_path.name} ---------')
    debug_utils.print_info(image_src, 'read image:')
    # image = skimage.transform.resize(
    #     image_src, model_trainer.model_input_image_size, anti_aliasing=True, order=1).astype(np.float32)
    image = image_src.astype(np.float32)
    image = cv2.resize(image, model_trainer.model_input_image_size, interpolation=cv2.INTER_AREA)
    debug_utils.print_info(image, 'resized image:')
    cam_threshold = 0.1
    male = image_path.name.startswith('m_')
    heatmap, age = model_trainer.generate_image_cam_overlay(image, male=male, cam_threshold=cam_threshold)
    print('result age:', age)
    cam_path = output_path / 'cam'
    cam_path.mkdir(exist_ok=True)
    skimage.io.imsave(str(cam_path / f'{image_path.stem}_heatmap_{model_trainer.model_name}.png'), heatmap)

    return image_path.name, male, age

    # cropped_to_cam = model_trainer.crop_image_to_cam(image_src, image, male=False, threshold=cam_threshold)
    # cropped_resized = skimage.transform.resize(
    #     cropped_to_cam, model_trainer.model_input_image_size, anti_aliasing=True, order=1).astype(np.float32)
    # skimage.io.imsave(str(OUTPUT_PATH / 'cam' / f'{image_path.stem}_cropped.png'),
    #                   image_utils.normalized_image(cropped_resized))


def crop_images_to_cam(model_trainer, csv_path, cropped_path):
    data_csv = pd.read_csv(str(csv_path))
    for index, csv_row in enumerate(data_csv.values):
        image_id, male, age = csv_row
        print(f'#{index} image_id: {image_id} male: {male} age: {age}')

        image_path = Path(r'D:\Projects\bone-age-models\data\images') / f'{image_id}.png'

        image_src = skimage.io.imread(str(image_path), as_gray=True)
        image = skimage.transform.resize(
            image_src, model_trainer.model_input_image_size, anti_aliasing=True, order=1).astype(np.float32)

        cam_threshold = 0.1
        cropped_to_cam = model_trainer.crop_image_to_cam(image_src, image, male=male, threshold=cam_threshold)
        cropped_resized = skimage.transform.resize(
            cropped_to_cam, model_trainer.model_input_image_size, anti_aliasing=True, order=1).astype(np.float32)
        skimage.io.imsave(str(cropped_path / image_path.name), image_utils.normalized_image(cropped_resized))


def analyze_our_data(model_trainer, data_path):
    # Convert file names to latin
    # data_path = TEMP_PATH / 'OurData'
    # latin_dir_path = latin_utils.create_dir_copy_with_latin_file_names(data_path)

    print('data_path', data_path)
    output_data = []
    for item_path in data_path.iterdir():
        print(item_path)
        item_suffix = item_path.suffix
        if item_suffix == '.jpg' or item_suffix == '.jpeg' or item_suffix == '.png':
            image_name, male, age = test_image(model_trainer, item_path, output_path=data_path)
            output_data.append([image_name, male, age])
    print('output_data', output_data)
    output_data_frame = pd.DataFrame(output_data, columns=['id', 'male', model_trainer.model_name])

    results_csv_path = data_path / 'results.csv'
    if results_csv_path.exists():
        results_data_frame = pd.read_csv(str(results_csv_path))
        # Remove male column, which exists in another DataFrame
        output_data_frame = output_data_frame.drop(columns='male')
        results_data_frame = results_data_frame.join(output_data_frame.set_index('id'), on='id')
    else:
        output_data_frame.insert(2, 'boneage', 0)  # Add column boneage with zero values (need to use our DataGenerator)
        results_data_frame = output_data_frame

    results_data_frame.to_csv(results_csv_path, index=False)


def count_children():
    data_frame = pd.read_csv(r'D:\Projects\bone-age-models\data\csv\all\train.csv')
    data = data_frame.to_numpy()
    print('len:', len(data))
    count_male = 0
    count_female = 0
    for index, data_row in enumerate(data):
        image_id, male, age = data_row
        if age <= 36:
            if male:
                count_male += 1
            else:
                count_female += 1
    print('male:', count_male)
    print('female:', count_female)


def main():
    model_trainer = trainer.DenseNetModelTrainer()

    # model_trainer.verify_generator(model_trainer.train_generator)
    model_trainer.run()
    exit()

    # model_trainer.load_model()
    # print(model_trainer.model_name)
    # model_trainer.model.summary(line_length=150)

    # model_trainer.create_model()
    # model_trainer.model.summary(line_length=150)
    # model_trainer.model.save(r'D:\Temp\TempBoneAgeModels\DenseNet_withInputShape.h5', include_optimizer=False)
    # exit()

    import keras
    # m1 = keras.models.load_model(r'D:\Temp\TempBoneAgeModels\DenseNet169_500x500_b7_AllImages.h5', compile=False)
    m2 = keras.models.load_model(r'D:\Temp\TempBoneAgeModels\DenseNet_withInputShape.h5', compile=False)
    m2.load_weights(r'D:\Temp\TempBoneAgeModels\DenseNet169_500x500_b7_AllImages.h5')
    m2.save(r'D:\Temp\TempBoneAgeModels\DenseNet_withInputShape___weighted.h5')
    exit()


    # data_frame_with_predictions = model_trainer.data_frame_with_predictions(constants.TEST_DATA_CSV_PATH)
    # data_frame_with_predictions.to_csv(OUTPUT_PATH / 'all_test_with_predictions.csv', index=False)
    # data_frame_with_predictions = model_trainer.data_frame_with_predictions(OUTPUT_PATH / 'all_valid_with_predictions.csv')
    # data_frame_with_predictions.to_csv(OUTPUT_PATH / 'all_valid_with_predictions2.csv', index=False)

    # Calculate MAE for test data
    # mae = model_trainer.calculate_mae(model_trainer.TEST_DATA_CSV_PATH)   #constants.TEST_DATA_CSV_PATH)
    # print('MAE', mae)

    # cropped_path = Path('C:/MyDiskBackup/Projects/BoneAge/Data/CroppedImages500PartKeepCloseAspectRatio')
    # crop_images_to_cam(model_trainer, constants.PART_TRAIN_DATA_CSV_PATH, cropped_path)
    # crop_images_to_cam(model_trainer, constants.PART_VALID_DATA_CSV_PATH, cropped_path)

    our_data_path = TEMP_PATH / 'OurData_latin'
    for image_dir in our_data_path.iterdir():
        # analyze_our_data(model_trainer, image_dir)

        # Analyze using combo model
        # model_trainer.IMAGE_DIR = image_dir
        # data_frame_with_combo_predictions = model_trainer.data_frame_with_predictions(image_dir / 'results.csv')
        # data_frame_with_combo_predictions.to_csv(image_dir / 'results.csv', index=False)

        # Format to readable age
        results = pd.read_csv(str(image_dir / 'results.csv'))
        results_data = results.to_numpy()
        columns_to_format = results_data[:, 3:]
        formatted_columns = np.core.defchararray.add((columns_to_format // 12).astype('|S10'),
                                                     np.full_like(columns_to_format, fill_value=' years ', dtype='|S10'))
        formatted_columns = np.core.defchararray.add(formatted_columns,
                                                     (columns_to_format % 12).astype('|S4'))
        formatted_columns = np.core.defchararray.add(formatted_columns,
                                                     np.full_like(columns_to_format, fill_value=' months ', dtype='|S10'))
        formatted_columns = np.core.defchararray.decode(formatted_columns, encoding='utf-8')

        results_data[:, 3:] = formatted_columns
        formatted_data_frame = pd.DataFrame(data=results_data, columns=results.columns)
        formatted_data_frame.to_csv(image_dir / 'formatted_results.csv', sep=';', index=False)


if __name__ == '__main__':
    main()
