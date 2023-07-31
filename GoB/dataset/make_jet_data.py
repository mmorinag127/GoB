import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import functools
import tensorflow.compat.v2 as tf
tf.config.experimental.set_visible_devices([], 'GPU')




import numpy as np
import glob
import time
from tqdm.auto import tqdm

from pdgid import PDGID
pdgid_table = PDGID()

def print_labels(l0, l1):
    s1 = pdgid_table.Name(l0)
    if l1 == (0, 0):
        s2 = 'other'
    elif l1 == (-1, -1):
        s2 = 'wrong'
    else:
        s2 = ''.join(pdgid_table.Name(k) for k in l1)
    return s1, s2

class Table:
    def __init__(self):
        self.table = {}
        self.total = 0
    
    def add(self, label):
        if label not in self.table.keys():
            self.table[label] = 0
        self.table[label] += 1
        self.total += 1
    
    def finalize(self):
        ret = {'total':self.total}
        print(f'total: {self.total: >10d} ')
        table = sorted(self.table.items())
        for key, val in table:
            print(f'{key: >16s} : {val: >8d}')
            ret[key] = val
        return ret

class Tables:
    def __init__(self, files):
        self.table = {key:Table() for key in files}
        self.table['all'] = Table()
        
    def add(self, label, filename):
        self.table['all'].add(label)
        self.table[filename].add(label)
    
    def finalize(self):
        self.table['all'].finalize()


def cast_str(l):
    key = l.numpy().decode('utf-8')
    return key

def cast_func(data):
    return tf.py_function(data)


from jet_data import my_tfrecord_data

def make_test_tfrecords(filename, test_var_list, n_max = 1000, compression = 'GZIP'):
    start = time.time()
    
    tfdata_handler = my_tfrecord_data(var_list = test_var_list)
    
    print(f'start writing')
    np.random.seed(127)
    
    def label_str(l0,l1):
        if l0 == 1:
            s0 = 'A' 
        elif l0 == 2:
            s0 = 'B'
        elif l0 == 3:
            s0 = 'C'
        else:
            s0 = 'Other'
        if l1 == 1:
            s1 = 'aa' 
        elif l1 == 2:
            s1 = 'bb'
        elif l1 == 3:
            s1 = 'cc'
        else:
            s1 = 'oo'
        return f'{s0} -> {s1}'
    
    # with tqdm(total = len(files), unit = 'file', ncols = 250, disable = False, bar_format="{desc}{percentage:3.0f}% {n_fmt: >5}/{total_fmt: >5} [{rate_fmt: >16}{postfix}]") as pbar:
    #     pbar.set_description('Making TFRecord')
    with tf.io.TFRecordWriter(filename+'.tfrecord', tf.io.TFRecordOptions(compression_type = compression)) as writer:
        
        for i in range(n_max):
            if i % 10000 == 0: 
                r = i / n_max *100.
                print(f'current : {i: 10d}({r: 3.2f}%)')
            
            data = {}
            N = np.random.randint(1,20,1)
            n = int(N[0])
            
            data['label0'] = list(np.random.randint(1,3,1))
            data['label1'] = list(np.random.randint(1,3,1))
            data['label']  = label_str(data['label0'], data['label1'])
            
            data['jet_f'] = list(np.random.randn(5))
            data['cnst_phi'] = list(np.random.randn(n))
            data['cnst_eta'] = list(np.random.randn(n))
            data['cnst_feature'] = np.random.randn(6, n)
            data['Pt'] = np.random.randn(1)
            
            data['cnst_n'] = n
            
            example = tfdata_handler.serialize(data)
            writer.write(example.SerializeToString())
            
            if n_max > 0 and i >= n_max:
                break
                #pbar.set_postfix(idx = idx)
                # pbar.update(1)
                
    stop = time.time()
    print(f'finish writing at {i}th sample')
    print(f'all time is {stop - start:.2f} sec')

def make_ds(filename, var_list, batch_size, compression = 'GZIP'):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_deterministic = False
    
    ds = tf.data.TFRecordDataset(filename + '.tfrecord', compression_type = compression)
    ds = ds.with_options(options)
    f_deserialize = my_tfrecord_data(var_list).make_deserialize()
    ds = ds.batch(batch_size)
    ds = ds.map(f_deserialize, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    #ds = ds.shuffle(batch_size*16)
    #ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    #yield from tfds.as_numpy(ds)
    return iter(ds)


def read_tf_record(file_pattern, var_list, batch_size = 1, compression = 'GZIP'):
    
    files = glob.glob(file_pattern)
    tab = Table()
    
    with tqdm(total = len(files), unit = 'file', ncols = 250, disable = False, bar_format="{desc}{percentage:3.0f}% {n_fmt: >5}/{total_fmt: >5} [{rate_fmt: >16}{postfix}]") as pbar:
        for filename in files:
            print(f'{filename} start')
            ds = make_ds(filename, var_list, batch_size = batch_size, compression=compression)
            
            idx = 0
            for data in ds:
                label = data['label']
                tab.add(label, filename)
                idx += 1
    table = tab.print_table()
    
    print(f'saving {filename}.meta.npz')
    np.savez_compressed(filename + '.meta.npz', **table)
    
    
    

if __name__ == '__main__':
    #filename = '/data/morinaga/work/GoB/data/large-R/data-2^10
    test_var_list = {'label0':'int', 'label1':'int', 'label':'str', 'jet_f':'float-ragged', 'cnst_eta':'float-ragged', 
                    'Pt':'float-fix',
                    'cnst_phi':'float-ragged', 'cnst_n':'int', 'cnst_feature':'bytes'}
    #test_var_list = {'label0':'int', 'label1':'int', 'jet_f':'float', 'cnst_eta':'float', 'cnst_phi':'float', 'cnst_n':'int'}
    #test_var_list = {'cnst_n':'int'}
    filename = '/data/morinaga/work/GoB/data/large-R/test-1000'
    compression = ''
    #make_test_tfrecords(filename, test_var_list, n_max = 10, compression=compression)
    #read_tf_record(filename, test_var_list, compression=compression)
    
    
    filename = '/data/morinaga/work/GoB/data/large-R/ver1/data-min_pt300.0GeV-2^22'
    compression = ''
    var_list = {'label0':'int', 'label1':'int', 'label2':'int', 'label':'str', 'n_cnst':'int',
                'pt':'float-ragged', 'eta':'float-ragged', 'phi':'float-ragged', #'p4':'float-ragged', 
                'track':'float-ragged', 'EMCal':'float-ragged', 'HCal':'float-ragged', 'd0':'float-ragged', 'd0_':'float-ragged', 'charge':'float-ragged',
                'Pt':'float-fix', 'Eta':'float-fix', 'Phi':'float-fix', 'E':'float-fix', 'M':'float-fix', 'n_charged':'float-fix', 'n_neutral':'float-fix'}
    read_tf_record(filename, var_list, batch_size = 1, compression = compression)
    



