import jax
import jax.numpy as jnp
import glob

import tensorflow.compat.v2 as tf
from .jet_data import my_tfrecord_data, make_jet_all
AUTOTUNE = tf.data.experimental.AUTOTUNE

class Dataset:
    def __init__(self, ds, skeleton = None):
        self.ds = ds
        self.it = map(lambda xs: jax.tree_map(lambda x: x._numpy(), xs), self.ds)
        self.skeleton = skeleton if skeleton is not None else next(iter(self.it))
        
    def make_bps(self, f, run_state):
        e = f.get_executable(run_state, self.skeleton)
        placement_specs = e.get_input_placement_specs()
        # print(placement_specs[1])
        return placement_specs[1]
    
    def make_iter(self, dataloader, f, run_state, n_prefetch):
        it = dataloader(self.it, self.make_bps(f, run_state), prefetch_size = n_prefetch)
        self.iter = iter(it)
        # self.iter = iter(self.it)
        
    def set_iter(self, it = None):
        if it is not None:
            self.iter = iter(it)
        else:
            self.iter = iter(self.it)

def load_minist_dataset(split, *, is_training, batch_size):
    import tensorflow_datasets as tfds
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return Dataset(ds)

def glob_datafiles(datafiles, phase, n_data):
    _files = []
    for datafile in datafiles:
        _files += glob.glob(f'{datafile}')
    
    if n_data.n_files > len(_files):
        print(f'#of data files({len(_files)}) is less than #of data_config({n_data.n_files})')
    _files = _files[:n_data.n_files]
    files = []
    
    for i, datafile in enumerate(_files):
        if 'train' in phase:
            if i % n_data.split_N != n_data.split_idx:
                files.append(datafile)
        else:
            if i % n_data.split_N == n_data.split_idx:
                files.append(datafile)
    return files

def make_dataset(phase, datafiles, batch_sizes, n_prefetch, compression, label_name, n_data, # split_N, split_idx, n_epochs, 
                var_list, data_def, xargs, cache = None, autotune = None, is_eval = False, is_n_device = False, **kwargs):
    batch_size = batch_sizes[phase] 
    if is_n_device:
        batch_size *= jax.local_device_count()
    
    _var_list = {}
    for key,val in xargs.items():
        _var_list[val] = var_list[val]
    
    for key, val in label_name.items():
        _var_list[key] = var_list[key]
    if is_eval:
        if 'glob' in _var_list:
            _var_list['glob'] = var_list['glob']
    
    if data_def['type'] == 'image':
        image_shape = [data_def['kwargs']['n_pixel'], data_def['kwargs']['n_pixel'], len(data_def['kwargs']['ch_list'])]
        f_deserialize = my_tfrecord_data(_var_list).make_deserialize(image = (batch_size, *image_shape))
    elif data_def['type'] == 'object':
        l = max(len(data_def['kwargs']['track_list']), len(data_def['kwargs']['cluster_list']))
        obj_x_shape = (batch_size, data_def['kwargs']['n_max_seq'], l)
        obj_m_shape = (batch_size, data_def['kwargs']['n_max_seq'], 1)
        obj_o_shape = (batch_size, data_def['kwargs']['n_max_seq'], 2)
        f_deserialize = my_tfrecord_data(_var_list).make_deserialize(obj_x = obj_x_shape, mask = obj_m_shape, obj_one_hot = obj_o_shape)
        
    elif data_def['type'] == 'pn_obj':
        l = max(len(data_def['kwargs']['track_list']), len(data_def['kwargs']['cluster_list']))
        n_max_seq = data_def['kwargs']['n_max_seq']
        feature_shape = (batch_size, n_max_seq, l)
        po_shape = (batch_size, n_max_seq, 2)
        mask_shape = (batch_size, n_max_seq, 1)
        glob_shape = (batch_size, data_def['kwargs']['n_glob'])
        f_deserialize = my_tfrecord_data(_var_list).make_deserialize(feature = feature_shape, position = po_shape, one_hot = po_shape, mask = mask_shape, glob=glob_shape)
        
    else:
        node_shape = (data_def['kwargs']['n_max_seq'], len(data_def['kwargs']['ch_list']))
        f_deserialize = my_tfrecord_data(_var_list).make_deserialize(node = (batch_size, *node_shape), mask = (batch_size, data_def['kwargs']['n_max_seq'], 1))
    
    if autotune is None:
        autotune = AUTOTUNE
    
    files = glob_datafiles(datafiles, phase, n_data)
    def cast_label(data):
        new_data = {}
        for key in data.keys():
            if key in label_name:
                new_key = label_name[key]
                new_data[new_key] = tf.cast(data[key], tf.float32)
            else:
                new_data[key] = data[key]
        return new_data
    
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_deterministic = False
    
    ds = tf.data.TFRecordDataset(files, compression_type = compression, num_parallel_reads=32)
    ds = ds.with_options(options)
    ds = ds.shuffle(buffer_size = 128*batch_size, seed = 0)
    
    ds = ds.batch(batch_size, drop_remainder = phase in ['train', 'test'])
    ds = ds.map(f_deserialize,  num_parallel_calls = autotune, name = 'deserialize_jet_data')
    ds = ds.map(cast_label,     num_parallel_calls = autotune, name = 'cast')
    
    if is_eval:
        pass
    else:
        if cache:
            print(f'cache will be made for ...')
            ds = ds.cache()
        if n_data.repeat:
            ds = ds.repeat()
    
    ds = ds.prefetch(n_prefetch, name = 'prefetch')
    
    return Dataset(ds, skeleton = None)

def make_dataset_top_data(phase, datafiles, batch_sizes, n_prefetch, compression, n_data, 
                var_list, data_def, xargs, cache = None, autotune = None, is_eval = False, is_n_device = False, n_max=None, **kwargs):
    batch_size = batch_sizes[phase] 
    if is_n_device:
        batch_size *= jax.local_device_count()
    
    
    l = len(data_def['kwargs']['track_list'])
    n_max_seq = data_def['kwargs']['n_max_seq']
    feature_shape = (batch_size, n_max_seq, l)
    position_shape = (batch_size, n_max_seq, 2)
    mask_shape = (batch_size, n_max_seq, 1)
    f_deserialize = my_tfrecord_data(var_list).make_deserialize(feature = feature_shape, position = position_shape, mask = mask_shape)
    
    if autotune is None:
        autotune = AUTOTUNE
    
    files = []
    for datafile in datafiles[phase]:
        files += glob.glob(f'{datafile}')
    
    def cast_label(data):
        new_data = {}
        for key in data.keys():
            if 'label' in key:
                new_data[key] = tf.cast(data[key], tf.float32)
            else:
                new_data[key] = data[key]
        return new_data
    
    def clip_track(data):
        # position, feature, mask, label
        for key in ['position', 'feature', 'mask']:
            d = data[key]
            data[key] = d[:, :n_max, :]
        return data
    
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_deterministic = False
    
    ds = tf.data.TFRecordDataset(files, compression_type = compression, num_parallel_reads=32)
    ds = ds.with_options(options)
    ds = ds.shuffle(buffer_size = 128*batch_size, seed = 0)
    
    ds = ds.batch(batch_size, drop_remainder = phase in ['train', 'test'])
    ds = ds.map(f_deserialize,  num_parallel_calls = autotune, name = 'deserialize_jet_data')
    ds = ds.map(cast_label,     num_parallel_calls = autotune, name = 'cast')
    if n_max is not None:
        ds = ds.map(clip_track, num_parallel_calls = autotune, name = 'clip_track')
    
    if is_eval:
        pass
    else:
        if cache:
            print(f'cache will be made for ...')
            ds = ds.cache()
        if n_data.repeat is not None:
            ds = ds.repeat()
    
    ds = ds.prefetch(n_prefetch, name = 'prefetch')
    
    return Dataset(ds, skeleton = None)




























