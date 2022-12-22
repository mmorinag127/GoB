

import functools
import math
# from pkgutil import read_code

import jax
import jax.numpy as jnp
import numpy as np


import tensorflow.compat.v2 as tf

#from .utils import prefetch_to_device, prepare_tf_data
from .flavor_data import deserialize_image

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

class DatasetSplit:
    def __init__(self, phase, info_file, N = 8, idx = 0):
        self.name = phase
        self.N = N
        self.idx = idx
        
        f = np.load(info_file, allow_pickle = True)
        
        if 'train' in self.name:
            self.length = math.floor((N - 1)/N*f['N_all'])
        else:
            self.length = math.floor(f['N_all']/N)
    def get_N_idx_str(self):
        return f'{self.idx}of{self.N}'
    
    def info(self, batch_size, n_device):
        n_step_per_epoch = math.ceil(self.length / (batch_size * n_device))
        data_info = {'n_step': n_step_per_epoch, 'length': self.length, 'batch_size': batch_size}
        return data_info
    
    def filter(self):
        if 'train' in self.name:
            def _filter(i, y):
                return i % self.N != self.idx
        else:
            def _filter(i, x):
                return i % self.N == self.idx
        return _filter

class Dataset:
    def __init__(self, ds, n_step, length, batch_size):
        self.ds = ds
        self.n_step = n_step
        self.length = length
        self.batch_size = batch_size
        self.it = map(lambda xs: jax.tree_map(lambda x: x._numpy(), xs), self.ds)
    
    def info(self):
        return {'n_step': self.n_step, 'length': self.length, 'batch_size': self.batch_size}
    
    def set_iter(self, it = None):
        if it is not None:
            self.it = iter(it)
        self.iter = iter(self.it)

def make_dataset(data_name, phase, split, batch_sizes, dtype, label_table, prop_table, cache = False, transpose = False, n_prefetch = 2, is_prop = False):
    
    batch_size = batch_sizes[phase]
    
    n_device = jax.local_device_count()
    
    if 'nominal' in data_name:
        data_dir = 'data/flavor'
        datafile = 'all-flavor-reco-32x32'
        datafiles = f'{data_dir}/{datafile}'
        compression = ''
        
    elif '2^20' in data_name:
        data_dir = 'data/flavor'
        datafile = 'all-flavor-reco-32x32-2^20'
        datafiles = f'{data_dir}/{datafile}'
        compression = ''
        
    elif '2^16' in data_name:
        data_dir = 'data/flavor'
        datafile = 'all-flavor-reco-32x32-2^16'
        datafiles = f'{data_dir}/{datafile}'
        compression = ''
        
    else:
        raise ValueError(f'{data_name} is not implemented yet...')
    
    preprocess_image = make_preprocess_image(datafiles + '.npz')
    preprocess_prop  = make_preprocess_prop(prop_table)
    preprocess_label = make_preprocess_label(label_table)
    ds_split = DatasetSplit(phase = phase, info_file = datafiles+'.npz', **split)
    
    
    
    
    
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_deterministic = False
    
    ds = tf.data.TFRecordDataset(datafiles+'.tfrecord', compression_type = compression)
    ds = ds.with_options(options)
    ds = ds.enumerate().filter(ds_split.filter(), name = ds_split.name)
    def de_enumerate(i,x):
        return x
    ds = ds.map(de_enumerate, num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'de_enumerate')
    
    # Only cache if we are reading a subset of the dataset.
    if cache and 'test' in phase:
        ds = ds.cache(f'data/cache/{datafile}.{ds_split.get_N_idx_str()}.test.cache.tfrecord')
    
    ds = ds.repeat()
    if 'train' in phase:
        ds = ds.shuffle(buffer_size = 16*batch_size*n_device, seed = 0)
    
    f_deserialize_image = functools.partial(deserialize_image, is_prop = is_prop)
    ds = ds.map(f_deserialize_image, num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'deserialize_image')
    
    ds = ds.map(preprocess_image, num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'preprocessing_image')
    if is_prop:
        ds = ds.map(preprocess_prop,  num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'preprocessing_prop')
    ds = ds.map(preprocess_label, num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'preprocessing_label')
    
    if transpose:
        def transpose_image(batch):
            # C H W -> H W C
            batch['image'] = tf.transpose(batch['image'], (1,2,0))
            return batch
        ds = ds.map(transpose_image, num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'transpose_image')
    
    if dtype != jnp.float32:
        def cast_image(batch):
            batch['image'] = tf.cast(batch['image'], tf.dtypes.as_dtype(dtype))
            return batch
        def cast_prop(batch):
            batch['prop'] = tf.cast(batch['prop'], tf.dtypes.as_dtype(dtype))
            return batch
        ds = ds.map(cast_image, num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'cast_image')
        if is_prop:
            ds = ds.map(cast_prop,  num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'cast_prop')
    
    
    ds = ds.batch(batch_size*n_device, drop_remainder = phase in ['train', 'test'])
    #ds = ds.prefetch(tf.data.experimental.AUTOTUNE, name = 'prefetch')
    
    ds = ds.prefetch(16, name = 'prefetch')
    
    
    #it = map(prepare_tf_data, ds)
    #it = prefetch_to_device(it, n_prefetch)
    #it = iter(ds)
    info = ds_split.info(batch_size, n_device)
    return Dataset(ds, **info)


