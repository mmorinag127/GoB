#from operator import is_
#from xml.sax.handler import feature_external_ges
import tensorflow.compat.v2 as tf
import numpy as np
import jax.numpy as jnp


class my_tfrecord_data:
    def __init__(self, var_list, is_shape = False):
        self.var_list = var_list
        self.is_shape = is_shape
    
    # def shape_feature(self, value):
    #     shape = np.array(value.shape, np.int32).tobytes()
    #     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape]))
    def shape_feature(self, value):
        shape = list(value.shape)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=shape)) 
    
    def bytes_feature(self, value):
        value = value.astype(np.float32).tobytes()
        if type(value) is not list:
            value = [value]
        val = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        return val
    
    def str_feature(self, value):
        value = value.encode('utf-8')
        if type(value) is not list:
            value = [value]
        val = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        return val
    
    def float_feature(self, value):
        if type(value) is not list:
            value = [value]
        val = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        return val
    
    def int_feature(self, value):
        if type(value) is not list:
            value = [value]
        val = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        return val
    
    def serialize(self, data):
        features = {}
        for key, dtype in self.var_list.items():
            # print(key, type(data[key]))
            # print(data[key])
            if 'bytes' in dtype:
                val = self.bytes_feature(data[key])
            elif 'str' in dtype:
                val = self.str_feature(data[key])
            elif 'float' in dtype:
                val = data[key]
                if type(val) is np.ndarray:
                    val = list(val)
                val = self.float_feature(val)
            elif 'int' in dtype:
                val = data[key]
                if type(val) is np.ndarray:
                    val = list(val)
                val = self.int_feature(val)
            else:
                raise ValueError(f'dtype : {dtype} of {key} is not supported!!')
            features[key] = val
        
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example
    
    def make_deserialize(self, **kwargs):
        var_list = self.var_list
        features = {}
        for key, dtype in var_list.items():
            if 'bytes' in dtype:
                features[key] = tf.io.FixedLenFeature([], tf.string)
            elif 'str' in dtype:
                features[key] = tf.io.FixedLenFeature([], tf.string)
            elif 'float-fix' in dtype:
                features[key] = tf.io.FixedLenFeature([], tf.float32)
            elif 'float-ragged' in dtype:
                features[key] = tf.io.RaggedFeature(tf.float32)
                #features[key] = tf.io.FixedLenFeature([], tf.float32)
            elif 'int' in dtype:
                features[key] = tf.io.FixedLenFeature([], tf.int64)
            else:
                raise ValueError(f'dtype : {dtype} of {key} is not supported!!')
        
        def deserialize(serialized_example):
            data = {}
            #example = tf.io.parse_single_example(serialized_example, features)
            example = tf.io.parse_example(serialized_example, features)
            for key, dtype in var_list.items():
                if 'bytes' in dtype:
                    value = tf.io.decode_raw(example[key], tf.float32, name = key)
                elif 'float' in dtype:
                    value = tf.cast(example[key], tf.float32)
                elif 'str' in dtype:
                    value = tf.cast(example[key], tf.string)
                elif 'int' in dtype:
                    value = tf.cast(example[key], tf.int64)
                
                
                if key in kwargs:
                    value = tf.reshape(value, kwargs[key])
                # print(f'{key} : {dtype} {value}, {value.shape}')
                data[key] = value
                
            return data
        return deserialize


def make_jet_property(glob_list):
    
    def jet_property(data):
        data['glob'] = tf.stack([data[key] for key in glob_list], axis = 1)
        return data
    return jet_property

def make_jet_label(table):
    #vtable = np.vectorize(lambda x:table[x])
    vtable = np.vectorize(lambda x:table.get(x, -1))
    def _cast_label(*data):
        label = data[0]
        key = label.numpy().astype('U13')
        ret = vtable(key)
        return ret
    
    def jet_label(data):
        label = tf.py_function(func = _cast_label, inp = [data['label']], Tout = tf.int32)
        data['label'] = label
        return data
    
    return jet_label

def make_jet_image(n_pixel, dR, ch_list, ch_last, is_np = False, eps = 1.0e-5):
    img_bin = np.linspace(-dR, +dR, n_pixel+1)
    img_bin = np.concatenate([np.array([-dR-eps]), img_bin, np.array([dR+eps])], axis = 0)
    img_bin = img_bin.reshape([-1, 1])
    im_shape = (n_pixel, n_pixel, len(ch_list)) if ch_last else (len(ch_list), n_pixel, n_pixel) 
    
    def binning_mask(val):
        val = val.reshape([1, -1])
        l_bin = np.less_equal(img_bin, val)
        r_bin = np.greater(img_bin, val)
        lr_bin = np.logical_and(l_bin[:-1,:], r_bin[1:, :])
        index = np.argmax(lr_bin, axis = 0) - 1
        mask = index >= 0
        return mask, index
    
    def _jet_image_np(eta, phi, *data):
        ch = np.stack(data, axis = 1)
        eta_mask, eta_idx = binning_mask(eta)
        phi_mask, phi_idx = binning_mask(phi)
        mask = eta_mask & phi_mask
        eta_idx = eta_idx[mask]
        phi_idx = phi_idx[mask]
        image = np.zeros(im_shape)
        if ch_last:
            image[eta_idx, phi_idx, :] += ch[mask, :]
        else:
            image[:, eta_idx, phi_idx] += ch[mask, :]
        return image
    if is_np:
        return lambda eta,phi,*data:{'image':_jet_image_np(eta,phi,*data)}
    
    _vjet_image_np = np.vectorize(_jet_image_np, otypes = [np.ndarray])
    
    def _jet_image(eta, phi, *data):
        eta = eta.numpy()
        phi = phi.numpy()
        data = [d.numpy() for d in data]
        image = _vjet_image_np(eta, phi, *data)
        image = np.stack(image, axis = 0)
        image = image.astype('float32')
        return tf.convert_to_tensor(image, dtype=tf.float32)
    
    def jet_image(data):
        image = tf.py_function(func = _jet_image, inp = [data['eta'], data['phi']] + [data[key] for key in ch_list], Tout = tf.float32)
        data['image'] = image
        return data
    
    return jet_image

def make_jet_node(ch_list, n_max_seq = 256): 
    def jet_node(data):
        node = np.stack([data[key] for key in ch_list])
        node = np.transpose(node, [1,0])
        n_cnst = data['n_cnst']
        if n_cnst > n_max_seq:
            mask = np.ones(shape=(n_max_seq, 1))
            node = node[:n_max_seq, :]
        else:
            n_remain = max(n_max_seq - n_cnst, 0)
            mask = np.concatenate([np.ones(shape=(n_cnst, 1),dtype=np.int32), np.zeros(shape=(n_remain, 1),dtype=np.int32)], axis=0)
            node = np.pad(node, ((0, n_remain), (0,0)))
            
        return {'node':node, 'mask':mask}
    return jet_node

def make_jet_object(track_list, cluster_list, n_max_seq):
    var_list = list(set(track_list+cluster_list))
    len_trk = len(track_list)
    len_cls = len(cluster_list)
    len_x = max([len_trk, len_cls])
    def jet_object(data):
        
        object_x = np.zeros([n_max_seq, len_x], dtype=np.float32)
        obj_one_hot = np.zeros([n_max_seq, 2], dtype=np.float32)
        mask = np.zeros([n_max_seq, 1], dtype=np.int32)
        
        trk_idx = np.squeeze(data['track'] > 0)
        cls_idx = np.squeeze(data['track'] == 0)
        
        len_trk_obj = trk_idx.shape[0] if trk_idx.ndim > 0 else 0
        len_cls_obj = cls_idx.shape[0] if cls_idx.ndim > 0 else 0
        
        if len_trk_obj > 0:
            trk_var = np.stack([data[key] for key in track_list])
            trk_var = np.transpose(trk_var, [1,0])
            object_x[:trk_idx.shape[0]][trk_idx, :len_trk] = trk_var[trk_idx, :]
            obj_one_hot[:trk_idx.shape[0]][trk_idx, 0] = 1
            mask[:trk_idx.shape[0]][trk_idx, 0] = 1
        
        if len_cls_obj > 0:
            cls_var = np.stack([data[key] for key in cluster_list])
            cls_var = np.transpose(cls_var, [1,0])
            object_x[:cls_idx.shape[0]][cls_idx, :len_cls] = cls_var[cls_idx, :]
            obj_one_hot[:cls_idx.shape[0]][cls_idx, 1] = 1
            mask[:cls_idx.shape[0]][cls_idx, 0] = 1
        
        return {'obj_x':object_x, 'obj_one_hot':obj_one_hot, 'obj_mask':mask}
    return jet_object

def make_pn_obj(track_list, cluster_list, n_max_seq, n_glob):
    var_list = list(set(track_list+cluster_list))
    len_trk = len(track_list)
    len_cls = len(cluster_list)
    len_x = max([len_trk, len_cls])
    
    def pn_obj(data):
        
        feature  = np.zeros([n_max_seq, len_x], dtype=np.float32)
        position = np.zeros([n_max_seq, 2], dtype=np.float32)
        one_hot = np.zeros([n_max_seq, 2], dtype=np.float32)
        mask = np.zeros([n_max_seq, 1], dtype=np.int32)
        
        trk_idx = np.squeeze(data['track'] > 0)
        cls_idx = np.squeeze(data['track'] == 0)
        
        len_trk_obj = trk_idx.shape[0] if trk_idx.ndim > 0 else 0
        len_cls_obj = cls_idx.shape[0] if cls_idx.ndim > 0 else 0
        
        if len_trk_obj > 0:
            trk_var = np.stack([data[key] for key in track_list])
            trk_var = np.transpose(trk_var, [1,0])
            trk_pos = np.stack([data[key] for key in ['eta', 'phi']])
            trk_pos = np.transpose(trk_pos, [1,0])
            
            feature[:trk_idx.shape[0]][trk_idx, :len_trk] = trk_var[trk_idx, :]
            position[:trk_idx.shape[0]][trk_idx, :2] = trk_pos[trk_idx, :]
            one_hot[:trk_idx.shape[0]][trk_idx, 0] = 1
            mask[:trk_idx.shape[0]][trk_idx, 0] = 1
        
        if len_cls_obj > 0:
            cls_var = np.stack([data[key] for key in cluster_list])
            cls_var = np.transpose(cls_var, [1,0])
            cls_pos = np.stack([data[key] for key in ['eta', 'phi']])
            cls_pos = np.transpose(cls_pos, [1,0])
            feature[:cls_idx.shape[0]][cls_idx, :len_cls] = cls_var[cls_idx, :]
            position[:cls_idx.shape[0]][cls_idx, :2] = cls_pos[cls_idx, :]
            one_hot[:cls_idx.shape[0]][cls_idx, 1] = 1
            mask[:cls_idx.shape[0]][cls_idx, 0] = 1
        
        return {'feature':feature, 'position':position, 'one_hot':one_hot, 'mask':mask}
    return pn_obj

def make_jet_all(label_table, glob_list, data_list, is_glob, **jet_data_kwargs):
    jet_image = make_jet_image(**jet_data_kwargs)
    jet_label = make_jet_label(label_table)
    jet_property = make_jet_property(glob_list)
    
    def jet_all(data):
        data = jet_image(data)
        # data['image'] = tf.zeros([32,32,3])
        data = jet_label(data)
        if is_glob:
            data = jet_property(data)
        new_data = {key:data[key] for key in data_list}
        return new_data
    return jet_all




























