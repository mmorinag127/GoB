#from operator import is_
#from xml.sax.handler import feature_external_ges
import tensorflow.compat.v2 as tf
import numpy as np


class my_tfrecord_data:
    def __init__(self, var_list):
        self.var_list = var_list
    
    def shape_feature(self, value):
        shape = np.array(value.shape, np.int32).tobytes()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape]))
    
    def bytes_feature(self, value):
        shape = self.shape_feature(value)
        val = value.astype(np.float32).tobytes()
        val = tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))
        return val, shape
    
    def float_feature(self, value):
        val = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        return val, None
    
    def int_feature(self, value):
        val = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        return val, None
    
    def serialize(self, data):
        features = {}
        for key, dtype in self.var_list:
            if 'bytes' in dtype:
                val, shape = self.bytes_feature(data[key])
            elif 'float' in dtype:
                val, shape = self.float_feature(data[key])
            elif 'int' in dtype:
                val, shape = self.int_feature(data[key])
            else:
                raise ValueError(f'dtype : {dtype} of {key} is not supported!!')
            features[key] = val
            if shape is not None:
                features[key+'_shape'] = shape
        
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()
    
    def make_deserialize(self):
        var_list = self.var_list
        def deserialize(serialized_example):
            features = {}
            for key, dtype in var_list:
                if 'bytes' in dtype:
                    features[key] = tf.io.FixedLenFeature([], tf.string)
                    features[f'{key}_shape'] = tf.io.FixedLenFeature([], tf.string)
                elif 'float' in dtype:
                    features[key] = tf.io.FixedLenFeature([], tf.float)
                elif 'int' in dtype:
                    features[key] = tf.io.FixedLenFeature([], tf.int64)
                else:
                    raise ValueError(f'dtype : {dtype} of {key} is not supported!!')
            
            data = {}
            example = tf.io.parse_single_example(serialized_example, features)
            for key, dtype in var_list:
                if 'bytes' in dtype:
                    shape = tf.io.decode_raw(example[f'{key}_shape'], tf.int32, name = f'{key}_shape')
                    value = tf.io.decode_raw(example[key], tf.float32, name = key)
                    value = tf.reshape(value, shape)
                elif 'float' in dtype:
                    value = tf.cast(example[key], tf.float)
                elif 'int' in dtype:
                    value = tf.cast(example[key], tf.int64)
                data[key] = value
            
            return data
        return deserialize









































