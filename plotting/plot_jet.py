import argparse, sys, os

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*10
colors = np.array(colors)

from color import nord_color
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax.numpy as jnp


def channel(idx):
    if   idx == 0 : return 'track'
    elif idx == 1 : return 'EMCal'
    elif idx == 2 : return 'HCal'
    elif idx == 3 : return 'sum of d0'
    elif idx == 4 : return 'pt'
    elif idx == 5 : return 'truth mass'
    return None
def get_label(l):
    if   l == 0: return 'None'
    elif l == 1: return 'd-jet'
    elif l == 2: return 'u-jet'
    elif l == 3: return 's-jet'
    elif l == 4: return 'c-jet'
    elif l == 5: return 'b-jet'
    elif l == 6: return 'g-jet'
    return 'None!'

def make_jet_image( img, label, figname, n_pixel = 32, n_channel = 5):
    fig = plt.figure(figsize=(8.0*n_channel, 8.0))
    gs = fig.add_gridspec(ncols = n_channel, nrows = 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    ax = ax.flatten()
    cmaps = ['nord_rb']*n_channel
    cmaps = ['RdBu_r']*n_channel
    
    for i in range(n_channel):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        
        max_val = np.max(img[i, :, :])
        if max_val == 0.0: 
            max_val = 0.1
        im = ax[i].imshow(img[i, :, :], cmap = cmaps[i], vmin = -1.0*max_val, vmax = max_val)
        ax[i].text(2, 2, f'{label}: {channel(i)}', fontsize = 32, bbox={'facecolor': 'white', 'pad': 10, 'alpha':0.0})
        #plt.colorbar(im, ax=ax[i])
        

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def make_jet_image2( imgs, label, figname, n_pixel = 32, n_channel = 5):
    fig = plt.figure(figsize=(8.0*n_channel, 8.0*len(imgs)))
    gs = fig.add_gridspec(ncols = n_channel, nrows = 2, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    #ax = ax.flatten()
    cmaps = ['nord_rb']*n_channel
    cmaps = ['RdBu_r']*n_channel
    
    for j, img in enumerate(imgs):
        for i in range(n_channel):
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].set_xticklabels([])
            ax[j, i].set_yticklabels([])
            
            max_val = np.max(img[i, :, :])
            if max_val == 0.0: 
                max_val = 0.1
            im = ax[j, i].imshow(img[i, :, :], cmap = cmaps[i], vmin = -1.0*max_val, vmax = max_val)
            ax[j, i].text(2, 2, f'{label}: {channel(i)}', fontsize = 32, bbox={'facecolor': 'white', 'pad': 10, 'alpha':0.0})
            #plt.colorbar(im, ax=ax[i])
        

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')
    
def make_all_jet_image(_imgs, figname, n_pixel = 32, n_channel = 5 ):
    imgs = deepcopy(_imgs)
    n_flavor = len(imgs.keys())
    fig = plt.figure(figsize=(8.0*n_flavor, 8.0*n_channel))
    
    
    width_ratios = [1.0]*n_flavor
    width_ratios[-1] = 1.05
    height_ratios = [1.0]*n_channel
    
    
    gs = fig.add_gridspec(ncols = n_flavor, nrows = n_channel, hspace = 0, width_ratios = width_ratios, height_ratios = height_ratios)
    ax = gs.subplots(sharex = True, sharey = True)
    
    cmaps = ['nord_rb', 'nord_rb', 'nord_rb', 'nord_rb', 'nord_rb']
    cmaps = ['RdBu_r']*n_channel
    
    vmin, vmax = [], []
    for j in range(n_channel):
        _vmin, _vmax = [], []
        for i, (key, img) in enumerate(imgs.items()) : 
            _vmin.append(np.min(img[j, :, :]))
            _vmax.append(np.max(img[j, :, :]))
        vmin.append(np.min(_vmin))
        vmax.append(np.max(_vmax))
    
    for i, (key, img) in enumerate(imgs.items()) : 
        for j in range(n_channel):
            max_val = max([abs(vmin[j]), abs(vmax[j])])
            ax[j,i].set_xticks([])
            ax[j,i].set_yticks([])
            ax[j,i].set_xticklabels([])
            ax[j,i].set_yticklabels([])
            ims = ax[j,i].imshow(img[j, :, :], cmap = cmaps[j], vmin = -1.0*max_val, vmax = max_val)
            if j == n_channel - 1:
                print(f'max_val is {max_val}')
                
            ax[j,i].text(2, 2, f'{key}: {channel(j)}', fontsize = 32, bbox={'facecolor': 'white', 'pad': 10, 'alpha':0.0})
            if i == n_flavor - 1:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax[j,i])
                cax = divider.append_axes("right", size="5%", pad=0.0)
                fig.colorbar(ims, cax = cax)
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def mkdir(dir):
    import os
    if not os.path.exists(f'{dir}'):
        os.makedirs(f'{dir}')

def plot_just_one_jet(opts):
    sys.path.append('GoB/')
    from dataset import dataset
    dataset_args = {
        'data_name': 'nominal',
        'batch_size': 1, 
        'split':{'N':20, 'idx':1},
        'n_prefetch': 2,
        'transpose': False,
        'cache': False,
        'prop_table': [0, 1, 4, 5],
        'lable_table': [0, 1, 2, 3, 4, 5, 6], # [None, d, u, s, c, b, g]
    }
    _, datasets = dataset.make_dataset(phase = 'test', **dataset_args, dtype = jnp.float32)
    
    
    fig_dir = f'figs/jet-image'
    mkdir(fig_dir)
    
    count = {}
    for idx, data in enumerate(datasets):
        image = data['image'][0][0]
        label = data['label'][0]
        print(image.shape, label.shape)
        label = get_label(int(label))
        
        if label not in count.keys():
            count[label] = 0
        else:
            count[label] += 1
        
        c = count[label]
        fig_name = f'{fig_dir}/jet-image-{label}-{c}.png'
        make_jet_image(image, label, fig_name)
        if idx >= 9:
            break

def main(opts, eps = 1e-7):
    data = np.load(opts.data_file, allow_pickle=True)
    
    image_raw = {}
    image_cor = {}
    
    flavors = ['d-jet', 'u-jet', 's-jet', 'c-jet', 'b-jet', 'g-jet']
    
    mean_image = np.zeros_like(data['mean_all'])
    std_image  = np.zeros_like(data['std_all'])
    
    mean_image = data['mean_all']
    std_image  = data['std_all']
    
    # for flavor in flavors:
    #     mean_image += data[f'mean_{flavor}']*data[f'N_{flavor}']/data[f'N_all']
    #     std_image  += data[f'std_{flavor}']*data[f'N_{flavor}']/data[f'N_all']
    
    
    for flavor in flavors:
        mean_image += data[f'N_{flavor}']/data[f'N_all'] * data[f'mean_{flavor}']
        std_image  += data[f'N_{flavor}']/data[f'N_all'] * data[f'std_{flavor}']
    
    for flavor in flavors:
        image_raw[flavor] = data[f'mean_{flavor}']
        image_cor[flavor] = (data[f'mean_{flavor}'] - mean_image)/(std_image + eps)
    
    make_all_jet_image({'mean':deepcopy(mean_image), 'std':deepcopy(std_image)}, figname = f'figs/{opts.data_ver}_mean_std.png', n_pixel = 32, n_channel = 5)
    make_all_jet_image({'mean':deepcopy(std_image), 'std':deepcopy(std_image)}, figname = f'figs/{opts.data_ver}_std.png', n_pixel = 32, n_channel = 5)
    make_all_jet_image({'mean':deepcopy(mean_image), 'std':deepcopy(mean_image)}, figname = f'figs/{opts.data_ver}_mean.png', n_pixel = 32, n_channel = 5)
    
    make_all_jet_image(image_raw, figname = f'figs/{opts.data_ver}_raw.png', n_pixel = 32, n_channel = 5)
    make_all_jet_image(image_cor, figname = f'figs/{opts.data_ver}_cor.png', n_pixel = 32, n_channel = 5)

def make_plots(): 
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    eps=1.0e-7
    labels = {'H->ss':0, 'H->cc':1, 'H->bb':2,
            'g->uu/dd':3, 'g->ss':4, 'g->cc':5, 'g->bb':6,
            'd/u-quark':7, 's-quark':8, 'c-quark':9, 'b-quark':10, 'gluon':11}
    
    data = np.load('all_image.npz', allow_pickle = True)
    
    fig_dir = 'figs/LargeR-jet-image-64'
    mkdir(fig_dir)
    sum1 = np.zeros(shape=(64,64,5))
    sum2 = np.zeros(shape=(64,64,5))
    weight = 0
    N_total = data['N_total']
    all_img = {}
    for label in labels.keys():
        m = data[f'{label}_mean']
        m = m.transpose(2,0,1)
        l = label.replace('->', '').replace('/','')
        #make_jet_image(m, label, figname = f'{fig_dir}/{l}.png', n_pixel=64, n_channel=5)
        w = N_total/data[f'{label}_total']
        print(label, data[f'{label}_total'], w)
        sum1 += data[f'{label}_sum1']*w
        sum2 += data[f'{label}_sum2']*w
        # print(data[f'{label}_sum1'][16:32,16:32,2])
        # print(data[f'{label}_sum1'][16:32,16:32,2]*w)
        # input('sum1')
        # print(data[f'{label}_sum2'][16:32,16:32,2])
        # input('sum2')
        weight += w
        all_img[label] = data[f'{label}_mean'].transpose(2,0,1)
    
    make_all_jet_image(all_img, figname = f'{fig_dir}/all_flavors.png', n_pixel=64, n_channel=5)
    mean = sum1/weight
    sigma = (sum2 - 2.0*mean*sum1 + weight*mean*mean)/weight
    #print(sum1[16:32,16:32,2])
    input('sum1')
    # print(sum2[16:32,16:32,2])
    # input('sum2')
    print(mean[16:32,16:32,2])
    input('mean')
    print('weight', weight)
    input('weight')
    print(sigma[16:32,16:32,2])
    input('sigma')
    sigma = np.sqrt(sigma)
    mean = mean.transpose(2,0,1)
    sigma = sigma.transpose(2,0,1)
    # print(mean[2])
    # input('mean')
    print(sigma[2])
    input('sigma')
    make_jet_image(mean,  'all', figname = f'{fig_dir}/all_mean.png',  n_pixel=64, n_channel=5)
    make_jet_image(sigma, 'all', figname = f'{fig_dir}/all_sigma.png', n_pixel=64, n_channel=5)
    
    for label in labels.keys():
        m = data[f'{label}_mean'].transpose(2,0,1)
        l = label.replace('->', '').replace('/','')
        m = (m - mean)/(sigma+eps)
        make_jet_image(m, label, figname = f'{fig_dir}/{l}_corr.png', n_pixel=64, n_channel=5)
        # print(m[2])
        # input('enter')
    






if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-dv', '--data_ver',   action = 'store', dest = 'data_ver',  default = '',  type = str)
    parser.add_argument('-df', '--data_file',  action = 'store', dest = 'data_file', default = '',  type = str)
    opts = parser.parse_args()
    
    
    #main(opts)
    #plot_just_one_jet(opts)
    make_plots()


















