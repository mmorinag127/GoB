import numpy as np
import tensorflow.compat.v2 as tf


def make_preprocess_image(file, eps = 1e-9):
    
    f = np.load(file, allow_pickle = True)
    mean = tf.convert_to_tensor(f['mean_all'], dtype = tf.float32)
    std  = tf.convert_to_tensor(f['std_all'],  dtype = tf.float32)
    
    def preprocess_image(example):
        image = example['image']
        image = (image - mean)/(std + eps)
        example['image'] = image
        return example
    return preprocess_image

def make_preprocess_label(table):
    table = tf.convert_to_tensor(table)
    def preprocess_label(example):
        label = example['label']
        label = tf.gather(table, label)
        label = tf.cast(label, tf.int32)
        # label = tf.reshape(label, ())
        example['label'] = label
        return example
    return preprocess_label

def make_preprocess_prop(table):
    def preprocess_prop(example):
        prop = tf.cast(tf.gather(example['prop'], table), tf.float32)
        example['prop'] = prop
        return example
    return preprocess_prop
