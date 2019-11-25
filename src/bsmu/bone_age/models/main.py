from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
import skimage.transform

from bsmu.bone_age.models import constants
from bsmu.bone_age.models import debug_utils
from bsmu.bone_age.models import image_utils
# from bsmu.bone_age.models.inception import trainer
from bsmu.bone_age.models.xception import trainer
# from bsmu.bone_age.models.stn import trainer


PROJECT_PATH = Path('../../../../')
TEMP_PATH = PROJECT_PATH / 'temp'
OUTPUT_PATH = PROJECT_PATH / 'output'


def test_image(model_trainer, image_path):
    image_src = skimage.io.imread(image_path, as_gray=True)
    print(f'\n\t\t--------- {image_path.name} ---------')
    debug_utils.print_info(image_src, 'read image:')
    image = skimage.transform.resize(
        image_src, model_trainer.model_input_image_size, anti_aliasing=True, order=1).astype(np.float32)
    debug_utils.print_info(image, 'resized image:')
    cam_threshold = 0.1
    heatmap, age = model_trainer.generate_image_cam_overlay(image, male=False, cam_threshold=cam_threshold)
    print('result age:', age)
    skimage.io.imsave(str(OUTPUT_PATH / 'cam' / f'{image_path.stem}_heatmap.png'), heatmap)

    cropped_to_cam = model_trainer.crop_image_to_cam(image_src, image, male=False, threshold=cam_threshold)
    cropped_resized = skimage.transform.resize(
        cropped_to_cam, model_trainer.model_input_image_size, anti_aliasing=True, order=1).astype(np.float32)
    skimage.io.imsave(str(OUTPUT_PATH / 'cam' / f'{image_path.stem}_cropped.png'),
                      image_utils.normalized_image(cropped_resized))


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


def main():
    model_trainer = trainer.XceptionModelTrainer()
    model_trainer.verify_generator(model_trainer.train_generator)
    model_trainer.run()

    exit()

    model_trainer.load_model()
    model_trainer.model.summary(line_length=150)

    # cropped_path = Path('C:/MyDiskBackup/Projects/BoneAge/Data/CroppedImages500PartKeepCloseAspectRatio')
    # crop_images_to_cam(model_trainer, constants.PART_TRAIN_DATA_CSV_PATH, cropped_path)
    # crop_images_to_cam(model_trainer, constants.PART_VALID_DATA_CSV_PATH, cropped_path)


    # TEST_DATA_PATH = TEMP_PATH / 'test_data'
    # for test_image_path in TEST_DATA_PATH.iterdir():
    #     if test_image_path.is_file():
    #         test_image(model_trainer, test_image_path)


if __name__ == '__main__':
    main()
