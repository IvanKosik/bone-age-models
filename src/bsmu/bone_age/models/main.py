from pathlib import Path

import numpy as np
import skimage.io
import skimage.transform

from bsmu.bone_age.models import debug_utils
from bsmu.bone_age.models.inception import trainer


PROJECT_PATH = Path('../../../../')
TEMP_PATH = PROJECT_PATH / 'temp'
OUTPUT_PATH = PROJECT_PATH / 'output'


def main():
    model_trainer = trainer.InceptionModelTrainer()
    # model_trainer.verify_generator(model_trainer.train_generator)
    # model_trainer.run()

    model_trainer.load_model()
    # model_trainer.model.summary(line_length=150)

    image = skimage.io.imread(TEMP_PATH / 'test_data/7108.png')
    debug_utils.print_info(image, 'read image:')
    image = skimage.transform.resize(
        image, model_trainer.model_input_image_size, anti_aliasing=True, order=1).astype(np.float32)
    debug_utils.print_info(image, 'resized image:')
    heatmap, age = model_trainer.generate_image_cam_overlay(image, male=False)
    print('result age:', age)
    skimage.io.imsave(str(OUTPUT_PATH / 'cam/heatmap.png'), heatmap)


if __name__ == '__main__':
    main()
