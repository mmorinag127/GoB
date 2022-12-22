

import functools
import math
# from pkgutil import read_code

import jax
import jax.numpy as jnp
import numpy as np


import tensorflow.compat.v2 as tf

#from .utils import prefetch_to_device, prepare_tf_data
from .jet_data import deserialize_image


class DatasetInfo:
    def __init__(self, n_device, phase, datafile, cache, compression = '', N = 8, idx = 0):
        self.n_device = n_device
        self.name = phase
        self.N = N
        self.idx = idx
        self.datafile = datafile
        self.cache = cache
        self.phase = phase
        self.datafile = datafile
        self.compression = compression
        
        f = np.load(self.datafile+'.npz', allow_pickle = True)
        
        if 'train' in self.name:
            self.length = math.floor((N - 1)/N*f['N_all'])
        else:
            self.length = math.floor(f['N_all']/N)
        self.n_step = math.ceil(self.length / (self.batch_size * n_device))
        
    
    def filter(self):
        if 'train' in self.name:
            def _filter(i, y):
                return i % self.N != self.idx
        else:
            def _filter(i, x):
                return i % self.N == self.idx
        return _filter
    
    def is_cache(self):
        if self.cache:
            return True
        return None
    
    def cache_name(self):
        N_idx_str = f'{self.idx}of{self.N}'
        return f'data/cache/{self.datafile}.{N_idx_str}.test.tfrecord'
    
    def dataset(self, ds, n_device):
        return Dataset(ds, self.n_step, self.length, self.batch_size)
        
class Dataset:
    def __init__(self, ds, n_step, length, batch_size):
        self.ds = ds
        self.n_step = n_step
        self.length = length
        self.batch_size = batch_size


def make_dataset(data_name, phase, split, batch_sizes, dtype, label_table, prop_table, cache = False, transpose = False, n_prefetch = 2, is_prop = False):
    
    
    preprocess_image = make_preprocess_image(datafiles + '.npz')
    preprocess_prop  = make_preprocess_prop(prop_table)
    preprocess_label = make_preprocess_label(label_table)
    ds_split = DatasetSplit(phase = phase, info_file = datafiles+'.npz', **split)
    
    

def make_tfds(datainfo, ):
    n_device = jax.local_device_count()
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_deterministic = False
    
    ds = tf.data.TFRecordDataset(datainfo.datafile+'.tfrecord', compression_type = datainfo.compression)
    ds = ds.with_options(options)
    ds = ds.enumerate().filter(datainfo.filter(), name = datainfo.name)
    def de_enumerate(i,x):
        return x
    ds = ds.map(de_enumerate, num_parallel_calls = tf.data.experimental.AUTOTUNE, name = 'de_enumerate')
    
    # Only cache if we are reading a subset of the dataset.
    if datainfo.is_cache():
        ds = ds.cache(datainfo.cache_name())
    
    if 'train' in datainfo.phase:
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size = 16*datainfo.batch_size*n_device, seed = 0)
    
    
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
    
    
    ds = ds.batch(datainfo.batch_size*n_device, drop_remainder = datainfo.phase in ['train', 'test'])
    #ds = ds.prefetch(tf.data.experimental.AUTOTUNE, name = 'prefetch')
    
    ds = ds.prefetch(datainfo.n_prefetch, name = 'prefetch')
    
    
    #it = map(prepare_tf_data, ds)
    #it = prefetch_to_device(it, n_prefetch)
    #it = iter(ds)
    
    return datainfo.dataset(ds)


