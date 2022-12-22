import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import functools
import tensorflow.compat.v2 as tf
tf.config.experimental.set_visible_devices([], 'GPU')


from jet_data import my_tfrecord_data

import numpy as np
import glob
import time
# from tqdm.auto import tqdm

from pdgid import PDGID
pdgid_table = PDGID()


def make_jet_data(filepath, data_type = 'data', labels = None):
    data = np.load(filepath)
    label0 = data['label0']
    label1 = data['label1']
    
    
    
    
    
    eta = data[f'{data_type}_eta']
    phi = data[f'{data_type}_phi']
    feature = data[f'{data_type}_feature']
    constituents = data[f'{data_type}_constituents']
    p4 = data[f'{data_type}_p4']
    positions = np.concatenate([eta, phi], axis=-1)
    
    return {'label0':label0, 'label1':label1, 'pos':positions, 'p4':p4, 'feature':constituents, 'global':feature}

class LabelHandler:
    def __init__(self):
        self.table = {}
    
    def add(self, l1, l2):
        key = self.label_string(l1,l2)
        if key not in self.table.keys():
            self.table[key] = 0
        self.table[key] += 1
        
    def label_string(self, l1, l2):
        s1 = pdgid_table.Name(l1)
        if l2 is None:
            s2 = 'None'
        elif l2 == (0,0):
            s2 = 'others'
        elif l2 == (-1,-1):
            s2 = 'wrong'
        else:
            s2 = ','.join(pdgid_table.Name(p) for p in l2)
        
        return f'{s1} -> {s2}'
    
    def finalize(self):
        print(f'---'*20)
        print(f'summary : ')
        total = 0
        for key, val in self.table.items():
            print(f'{key: >16s} : {val: >10d}')
            total += val
        key = 'total'
        print(f'{key: >16s} : {total: >10d}')
        
    

def make_jet_tfrecords(filename, filepath, data_type, n_max = 1000, compression = 'GZIP', valid_labels = None ):
    start = time.time()
    
    print(f'start glob files')
    files = glob.glob(filepath)
    n_file = len(files)
    print(f'start permutation {n_file}files')
    rng = np.random.default_rng(seed = 0)
    perms = rng.permutation(len(files))
    
    
    var_list = ['label0':'int', 'label1':'int', 'pos':'float', 'global':'float', 'p4':'float', 'feature':'float']
    
    tfdata_handler = my_tfrecord_data(var_list = var_list)
    lh = LabelHandler()
    
    print(f'start writing')
    
    
    # with tqdm(total = len(files), unit = 'file', ncols = 250, disable = False, bar_format="{desc}{percentage:3.0f}% {n_fmt: >5}/{total_fmt: >5} [{rate_fmt: >16}{postfix}]") as pbar:
    #     pbar.set_description('Making TFRecord')
    with tf.io.TFRecordWriter(filename, tf.io.TFRecordOptions(compression_type = compression)) as writer:
        
        for i, idx in enumerate(perms):
            if i % 10000: 
                r = i / n_file *100.
                print(f'current : {i: 10d}({r: 3.2f}%)')
            data = make_jet_data(files[idx], data_type = data_type, labels = valid_labels)
            if data is None:
                continue
            
            example = tfdata_handler.serialize(data)
            writer.write(example)
            l1, l2 = int(data['label0']), tuple(data['label1'])
            lh.add(l1,l2)
            
            
            if n_max > 0 and i >= n_max:
                break
                #pbar.set_postfix(idx = idx)
                # pbar.update(1)
                
    lh.finalize()
    stop = time.time()
    print(f'all time is {stop - start:.2f} sec')



if __name__ == '__main__':
    filename = '/data/morinaga/work/GoB/data/flavor/all-flavor-reco-32x32-2^20.tfrecord'
    #filename = '/data/morinaga/work/GoB/data/flavor/all-flavor-reco-32x32-2^16.tfrecord'
    make_jet_tfrecords(
        filename = filename,
        filepath = '/home/morinaga/work/data/flavor/2022-06-26/?-jet/*.npz', 
        data_type = 'reco',
        is_image = True,
        dR = 0.4, 
        n_pixel = 32,
        n_max = 2**20,
        #n_max = 2**16,
        compression = '',
        valid_labels = [1, 2, 3, 4, 5, 6]
        )
    
    
