
import functools
import tensorflow.compat.v2 as tf
tf.config.experimental.set_visible_devices([], 'GPU')

from flavor_data import get_binning_mask, serialize_image, deserialize_image#, serialize_graph

import numpy as np
import glob
import time
from tqdm.auto import tqdm


def make_jet_image(filepath, image_type, dR = 0.4, n_pixel = 32, labels = None):
    data = np.load(filepath)
    label = data['label']
    if labels is not None:
        if label not in labels:
            return None, None, None
    
    eta = data[f'{image_type}_eta']
    phi = data[f'{image_type}_phi']
    channel = data[f'{image_type}_channel']
    
    prop = data['prop']
    
    mask, eta_idx, phi_idx = get_binning_mask(eta, phi, n_pixel = n_pixel, dR = dR)
    
    eta_idx = eta_idx[mask]
    phi_idx = phi_idx[mask]
    channel = channel[:, mask]
    image = np.zeros((channel.shape[0], n_pixel, n_pixel))
    image[:, eta_idx, phi_idx] += channel[:]
    
    return image, prop, label

def make_jet_graph(filepath, image_type, labels = None):
    data = np.load(filepath)
    label = data['label']
    if labels is not None:
        if label not in labels:
            return None, None, None
    
    eta = data[f'{image_type}_eta']
    phi = data[f'{image_type}_phi']
    channel = data[f'{image_type}_channel']
    p4 = data[f'{image_type}_p4']
    prop = data['prop']
    
    points = np.concatenate([eta, phi], axis=-1)
    
    jet_pt = prop[:, 0] # pt, eta, phi, e, m, #of charged, #of neutral
    jet_E = prop[:, 3]
    pt = p4[:, 0]
    E = p4[:, 3]
    
    dR = np.sqrt(eta**2 + phi**2)
    logpt = np.log(pt)
    logE = np.log(E)
    logpt2 = np.log(pt/jet_pt)
    logE2 = np.log(E/jet_E)
    
    mask_track  = np.float(channel[:, 0] > 0.)
    mask_EMcalo = np.float(channel[:, 1] > 0.)
    mask_HCalo  = np.float(channel[:, 2] > 0.)
    features = np.concatenate([logpt, logE, logpt2, logE2, dR, mask_track, mask_EMcalo, mask_HCalo])
    
    return points, features, labels

def make_image_tfrecords(filename, filepath, image_type, is_image, dR = 0.4, n_pixel = 32, n_max = 1000, compression = 'GZIP', valid_labels = None ):
    start = time.time()
    
    print(f'start glob files')
    files = glob.glob(filepath)
    print(f'start permutation {len(files)}files')
    rng = np.random.default_rng(seed = 0)
    perms = rng.permutation(len(files))
    
    labels = { key:0 for key in [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15,]}
    print(f'start writing')
    
    
    with tqdm(total = len(files), unit = 'file', ncols = 250, disable = False, bar_format="{desc}{percentage:3.0f}% {n_fmt: >5}/{total_fmt: >5} [{rate_fmt: >16}{postfix}]") as pbar:
        pbar.set_description('Making TFRecord')
        with tf.io.TFRecordWriter(filename, tf.io.TFRecordOptions(compression_type = compression)) as writer:
            
            for i, idx in enumerate(perms):
                if is_image:
                    image, prop, label = make_jet_image(files[idx], image_type = image_type, dR = dR, n_pixel = n_pixel, labels = valid_labels)
                    if image is None:
                        continue
                    example = serialize_image(image, prop, label)
                else:
                    points, features, label = make_jet_graph(files[idx], image_type = image_type, labels = valid_labels)
                    if points is None:
                        continue
                    example = serialize_graph(points, features, label)
                
                writer.write(example)
                l = int(label)
                labels[l] += 1
                if n_max > 0 and i >= n_max:
                    break
                #pbar.set_postfix(idx = idx)
                pbar.update(1)
                
    jets = {1:'d-jet', 2: 'u-jet', 3: 's-jet', 4:'c-jet', 5:'b-jet', 6:'g-jet', 11:'gd-jet', 12:'gu-jet', 13:'gs-jet', 14:'gc-jet', 15:'gb-jet', }
    for key, label in jets.items():
        print(f'{label: >10s}: {labels[key]: >8d}')
    
    stop = time.time()
    print(f'all time is {stop - start:.2f} sec')


def make_image_tfds(files, batch_size, compression = 'GZIP'):
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    
    ds = tf.data.TFRecordDataset(files, compression_type = compression)
    
    functools.partial(deserialize_image, is_prop = True)
    
    ds = ds.map(deserialize_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    #ds = ds.shuffle(batch_size*16)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    #yield from tfds.as_numpy(ds)
    return iter(ds)

class Average:
    def __init__(self, sample, shape):
        self.sample = sample
        self.x1 = np.zeros(shape, dtype = np.float32)
        self.x2 = np.zeros(shape, dtype = np.float32)
        self.N = 0
    
    def add(self, x):
        self.x1 += np.sum(x,    axis = 0)
        self.x2 += np.sum(x**2, axis = 0)
        self.N += x.shape[0]
    
    def calc(self):
        self.mean = self.x1 / self.N
        self.std  = self.x2 - self.N*self.mean*self.mean
        self.std  = np.sqrt(self.std/self.N)
        
    def add_other(self, other, weight):
        self.x1 += other.x1 * weight
        self.x2 += other.x2 * weight * weight
        self.N  += other.N

def check_label(filename, batch_size, compression):
    ds = make_image_tfds(filename+'.tfrecord', batch_size, compression = compression)
    labels = {i:0 for i in range(20)}
    idx = 0
    
    for data in ds:
        label = data['label']
        labels[int(label)] += 1
        
        idx += 1
        
        if idx >= 1000:
            break
        
        
    for k, v in labels.items():
        print(f'{k} : {v}')


def read_tfrecord(filename, batch_size, compression):
    print(f'{filename} start')
    
    shape = (5, 32, 32)
    labels = [1,2,3,4,5,6]
    jets = {1:'d-jet', 2: 'u-jet', 3: 's-jet', 4:'c-jet', 5:'b-jet', 6:'g-jet', 11:'gd-jet', 12:'gu-jet', 13:'gs-jet', 14:'gc-jet', 15:'gb-jet', }
    avgs = [Average(jets[l], shape ) for l in labels]
    
    
    ds = make_image_tfds(filename+'.tfrecord', batch_size, compression = compression)
    i = 0
    for data in ds:
        if i % 1000 == 0:
            print(f'{i: >6d}th')
        
        i += 1
        
        image, label = data['image'], data['label']
        for l, avg in zip(labels, avgs):
            avg.add(image[label == l])
            pass
        # if i > 10000 :
        #     break
        pass
    
    n_samples = {'all':0}
    
    for l, avg in zip(labels, avgs):
        jet_l = jets[l]
        n_samples[jet_l] = avg.N
        n_samples['all'] += avg.N
    
    all_sum = Average('all', shape)
    
    for avg in avgs:
        all_sum.add_other( avg, avg.N / n_samples['all'])
    
    
    means = {}
    stds  = {}
    for avg in avgs:
        avg.calc()
        means[f'mean_{avg.sample}'] = avg.mean
        stds[f'std_{avg.sample}'] = avg.std
    
    
    n_samples = {'N_'+key:np.array([val]) for key, val in n_samples.items() }
    all_sum.calc()
    means['mean_all'] = all_sum.mean
    stds['std_all'] = all_sum.std
    
    np.savez_compressed(filename + '.npz', **n_samples, **means, **stds )
    
    for key, val in n_samples.items():
        p = val/n_samples['N_all']*100.0
        print(key, float(val), float(p))

def make_dataset():
    
    filename = '/data/morinaga/work/GoB/data/flavor/all-flavor-reco-32x32-max1000.tfrecord'
    make_image_tfrecords(
        filename = filename,
        filepath = '/home/morinaga/work/data/flavor/2022-06-26/?-jet/*.npz', 
        image_type = 'reco',
        is_image = True,
        dR = 0.4, 
        n_pixel = 32,
        n_max = 1000,
        compression = '',
        valid_labels = [1, 2, 3, 4, 5, 6]
        )

if __name__ == '__main__':
    
    #make_dataset()
    
    read_tfrecord(filename = '/data/morinaga/work/GoB/data/flavor/all-flavor-reco-32x32-max1000', batch_size = 128, compression = '')
    #check_label(filename = '/data/morinaga/work/GoB/data/flavor/all-flavor-reco-32x32', batch_size = 1, compression = '')
    

    # rng = np.random.default_rng(seed = 0)
    # perms = rng.permutation(10)
    # print(perms)
    
    
    