import cv2
import numpy as np

from bsmu.bone_age.models import keras_utils, image_utils


def generate_cam_batch(input_batch, model, input_image_layer_name: str, input_male_layer_name: str,
                       output_age_layer_name: str, output_conv_layer_name: str, output_pooling_layer_name: str):
    output_function = keras_utils.model_output_function(
        model, [input_image_layer_name, input_male_layer_name],
        [output_age_layer_name, output_conv_layer_name, output_pooling_layer_name])
    output_age_batch, output_conv_batch, output_pooling_batch = output_function(input_batch)
    batch_size = output_age_batch.shape[0]
    cam_batch = np.zeros(shape=(batch_size, *output_conv_batch.shape[1:3]), dtype=np.float32)
    for sample_index in range(batch_size):
        output_conv = output_conv_batch[sample_index]
        output_pooling = output_pooling_batch[sample_index]
        cam = calculate_cam(output_conv, output_pooling)
        cam_batch[sample_index, ...] = cam
    return cam_batch, output_age_batch


def overlay_cam(image, cam, cam_threshold: float = 0.05):
    """
    :param image: source image
    :param cam: normalized to [0, 1] class activation map
    :param cam_threshold: heatmap will not be displayed in regions with cam values below this value
    :return: image with cam overlay
    """
    image_size = image.shape[:2]
    cam = cv2.resize(cam, image_size, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(image_utils.normalized_8U_image(cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < cam_threshold)] = 0
    heatmap = heatmap[:, :, ::-1]  # Invert channels order (OpenCV BGR to scikit-image RGB)
    image = image_utils.normalized_8UC3_image(image)
    overlay_result = np.zeros_like(image)
    cv2.addWeighted(heatmap, 0.5, image, 1, 0, dst=overlay_result)
    return overlay_result


def calculate_cam(conv, pooling):
    cam = np.copy(conv)
    for feature_map_index in range(cam.shape[2]):
        cam[..., feature_map_index] *= pooling[feature_map_index]
    cam = np.mean(cam, axis=-1)
    cam = image_utils.normalized_image(cam)
    return cam
