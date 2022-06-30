

import tensorflow.compat.v2 as tf
from typing import Tuple

def _normalize_image(image, mean_rgb, stddev_rgb):
    """Normalize the image to zero mean and unit variance."""
    image -= tf.constant(mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)
    return image


def _distorted_bounding_box_crop(
    image_bytes: tf.Tensor,
    *,
    jpeg_shape: tf.Tensor,
    bbox: tf.Tensor,
    min_object_covered: float,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float],
    max_attempts: int,
    ) -> tf.Tensor:
    """Generates cropped_image using one of the bboxes randomly distorted."""
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(jpeg_shape,
                                                                    bounding_boxes=bbox,
                                                                    min_object_covered=min_object_covered,
                                                                    aspect_ratio_range=aspect_ratio_range,
                                                                    area_range=area_range,
                                                                    max_attempts=max_attempts,
                                                                    use_image_if_no_bounding_boxes=True)
    
    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image

def _decode_and_random_crop(image_bytes: tf.Tensor) -> tf.Tensor:
    """Make a random crop of 224."""
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = _distorted_bounding_box_crop(image_bytes, jpeg_shape=jpeg_shape, bbox=bbox, min_object_covered=0.1, aspect_ratio_range=(3 / 4, 4 / 3), area_range=(0.08, 1.0), max_attempts=10)
    if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
        # If the random crop failed fall back to center crop.
        image = _decode_and_center_crop(image_bytes, jpeg_shape)
    return image

def _decode_and_center_crop(image_bytes, jpeg_shape = None) -> tf.Tensor:
    """Crops to center of image with padding then scales."""
    if jpeg_shape is None:
        jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = jpeg_shape[0]
    image_width = jpeg_shape[1]

    padded_center_crop_size = tf.cast(((224 / (224 + 32)) * tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width, padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image

def _preprocess_image(image_bytes, dtype, is_training, mean_rgb, stddev_rgb):
    """Returns processed and resized images."""
    if is_training:
        #image = _decode_and_random_crop(image_bytes)
        #image = tf.image.random_flip_left_right(image)
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        
    else:
        #image = _decode_and_center_crop(image_bytes)
        image = tf.image.decode_jpeg(image_bytes, channels=3)
    assert image.dtype == tf.uint8
    # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
    # clamping overshoots. This means values returned will be outside the range
    # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
    image = tf.image.resize(image, [224, 224], tf.image.ResizeMethod.BICUBIC)
    image = _normalize_image(image, mean_rgb, stddev_rgb)
    image = tf.image.convert_image_dtype(image, dtype = dtype)
    return image


def _preprocess_image_train(image_bytes, dtype, mean_rgb, stddev_rgb):
    """Returns processed and resized images."""
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    assert image.dtype == tf.uint8
    # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
    # clamping overshoots. This means values returned will be outside the range
    # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
    image = tf.image.resize(image, [224, 224], tf.image.ResizeMethod.BICUBIC)
    image = _normalize_image(image, mean_rgb, stddev_rgb)
    image = tf.image.convert_image_dtype(image, dtype = dtype)
    return image

def _preprocess_image_test(image_bytes, dtype, mean_rgb, stddev_rgb):
    """Returns processed and resized images."""
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    assert image.dtype == tf.uint8
    # NOTE: Bicubic resize (1) casts uint8cat to float32 and (2) resizes without
    # clamping overshoots. This means values returned will be outside the range
    # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
    image = tf.image.resize(image, [224, 224], tf.image.ResizeMethod.BICUBIC)
    image = _normalize_image(image, mean_rgb, stddev_rgb)
    image = tf.image.convert_image_dtype(image, dtype = dtype)
    return image

