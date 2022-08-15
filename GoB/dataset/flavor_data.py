from operator import is_
import tensorflow.compat.v2 as tf
import numpy as np

def _get_binning_mask(val, n_pixel, dR, eps):
    bin = np.linspace(-dR, dR, n_pixel)
    bin = np.insert(bin, 0, [-dR - eps])
    bin = np.insert(bin, n_pixel+1, [dR + eps])
    idx = np.searchsorted(bin, val, side = 'right')
    idx = idx - 1
    mask = (idx >= 0) & (idx<n_pixel)
    return mask, idx

def get_binning_mask(eta, phi, n_pixel, dR, eps = 1.0e-4):
    mask_eta, eta_idx = _get_binning_mask(eta, n_pixel, dR, eps)
    mask_phi, phi_idx = _get_binning_mask(phi, n_pixel, dR, eps)

    mask = mask_eta & mask_phi
    return mask, eta_idx, phi_idx

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_sample(image, prop, label):
    image_list = _bytes_feature(image.astype(np.float32).tobytes())
    prop_list = _bytes_feature(prop.astype(np.float32).tobytes())
    label_list = _int64_feature(label)
    ishape_list = _bytes_feature(np.array(image.shape, np.int32).tobytes())
    pshape_list = _bytes_feature(np.array(prop.shape,  np.int32).tobytes())
    
    proto = tf.train.Example(features=tf.train.Features(feature={
        'ishape' : ishape_list,
        'pshape' : pshape_list,
        'image': image_list, 
        'prop': prop_list, 
        'label': label_list 
    }))
    return proto.SerializeToString()

def deserialize(serialized_example, is_prop = False):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'ishape': tf.io.FixedLenFeature([], tf.string),
            'pshape': tf.io.FixedLenFeature([], tf.string),
            'image' : tf.io.FixedLenFeature([], tf.string),
            'prop'  : tf.io.FixedLenFeature([], tf.string),
            'label' : tf.io.FixedLenFeature([], tf.int64),
        })
    
    ishape = tf.io.decode_raw(features['ishape'], tf.int32,   name = 'ishape')
    pshape = tf.io.decode_raw(features['pshape'], tf.int32,   name = 'pshape')
    image  = tf.io.decode_raw(features['image'],  tf.float32, name = 'image')
    if is_prop:
        prop   = tf.io.decode_raw(features['prop'],   tf.float32, name = 'prop')
        prop  = tf.reshape(prop, pshape)
    
    image = tf.reshape(image, ishape)
    
    label = tf.cast(features['label'],  tf.int32)
    
    if is_prop:
        data = {'image':image, 'prop':prop, 'label':label}
    else:
        data = {'image':image, 'label':label}
    return data

