from pathlib import Path

import cv2
import numpy as np
import skimage.io
import skimage.transform

from bsmu.bone_age.models import image_utils
from bsmu.bone_age.models import debug_utils


OUTPUT_SIZE = (500, 500)
SRC_IMAGE_PATH = Path(r'D:\Projects\bone-age-models\data\images')
DST_IMAGE_PATH = Path(r'C:\MyDiskBackup\Projects\BoneAge\Data\SmallImages500')


def padded_image(image):
    """Add zero-padding to make square image (original image will be in the center of paddings)"""
    pads = np.array(image.shape).max() - image.shape
    before_pads = np.ceil(pads / 2).astype(np.int)
    after_pads = pads - before_pads
    pads = tuple(zip(before_pads, after_pads))
    image = np.pad(image, pads, mode='constant')
    return image, pads


def convert_images(src_path: Path, dst_path: Path):
    for index, image_path in enumerate(src_path.iterdir()):
        print(index)
        image = skimage.io.imread(str(image_path))
        image = padded_image(image)[0]

        image = image.astype(np.float64)
        # image = skimage.transform.resize(image, OUTPUT_SIZE, anti_aliasing=True, order=1).astype(np.float32)
        # image = skimage.transform.resize(image, OUTPUT_SIZE, anti_aliasing=True, order=2).astype(np.float32)
        image = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        # debug_utils.print_info(image, 'resized')

        image = image_utils.normalized_image(image)
        # debug_utils.print_info(image, 'normalized')

        skimage.io.imsave(str(dst_path / image_path.name), image)


def main():
    convert_images(SRC_IMAGE_PATH, DST_IMAGE_PATH)


if __name__ == '__main__':
    main()
