import tensorflow.compat.v2 as tf
import numpy as np

class PhysicsLabel:
    l1_labels = {1:'d', 2:'u', 3:'s', 4:'c', 5:'b', 6:'t', 21:'g', 23:'Z', 24:'W', 25:'H'}
    l2_labels = {1:'dd', 2:'uu', 3:'ss', 4:'cc', 5:'bb', 0:'others'}
    def __init__(self, label_type, label0_table, label1_table):
        self.label_type = label_type
        self.label0_valid = label0_table
        self.label1_valid = label1_table
        pass
    
    def make_label_translate(self):
        table0 = self.label0_table
        table1 = self.label1_table
        
        def label_translate(data):
            label0 = data['label0']
            label1 = data['label1']
            label1 = tf.cast(tf.gather(table0, label0), tf.int32)
            label0 = tf.cast(tf.gather(table1, label1), tf.int32)
            data['label0'] = label0
            data['label1'] = label1
            return data
        return label_translate
    
    def physics_label(self, label):
        l1, l2 = int(label[0]), int(label[1])
        if l1 not in self.l1_labels.keys() or l2 not in self.l2_labels.keys():
            return None
        
        return f'{self.l1_labels[l1]} -> {self.l2_labels[l2]}'


var_list = {'label0':'bytes', 'label1':'bytes', 'pos':'bytes', 'p4':'bytes', 'constituents':'bytes',
            'Pt':'bytes', 'Eta':'bytes', 'Phi':'bytes', 'E':'bytes', 'M':'bytes', 'n_charged':'bytes', 'n_neutral':'bytes'}
def make_jet_image(n_pixel, dR, channels):
    
    def binning_mask(val, eps = 1.0e-5):
        bin = np.linspace(-dR, dR, n_pixel)
        bin = np.insert(bin, 0, [-dR - eps])
        bin = np.insert(bin, n_pixel+1, [dR + eps])
        idx = np.searchsorted(bin, val, side = 'right')
        idx = idx - 1
        mask = (idx >= 0) & (idx <n_pixel)
        return mask, idx
    
    
    
    
    
    def jet_image(pos, constituents):
        eta, phi = pos[..., 0], pos[..., 1]
        eta_mask, eta_idx = binning_mask(eta)
        phi_mask, phi_idx = binning_mask(phi)
        mask = eta_mask & phi_mask
        eta_idx = eta_idx[mask]
        phi_idx = phi_idx[mask]
        
        constituents = constituents[mask]
        
        
        
        
    