

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*10
colors = np.array(colors)

from mpl_toolkits.mplot3d import axes3d

import torch


import utils
from color import nord_color

def mkdir(dir):
    if not os.path.exists(f'{dir}'):
        os.makedirs(f'{dir}')

def make_2d_contorf( X, Y, Z, n_level, figname, is_face = False, z_limit = None):
    fig = plt.figure(figsize=(8.0, 6.0))
    gs = fig.add_gridspec(ncols = 1, nrows = 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    
    if z_limit is not None:
        Z[Z > z_limit] = z_limit
    
    # l1 = np.linspace(np.min(Z), np.min(Z)*1.5, 100)
    # l2 = np.linspace(np.min(Z)*1.5, np.max(Z), n_level)
    # levels = np.concatenate([l1, l2[1:]])
    
    levels = np.linspace(np.min(Z), np.max(Z), n_level)
    #levels = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    if False:
        levels = []
        sorted_Z = np.sort(Z.ravel())
        length_Z = len(sorted_Z)
        
        for i in range(n_level):
            idx = int(i/n_level*length_Z)
            levels.append(sorted_Z[idx])
        #print(levels)
    cmap = plt.cm.viridis
    cmap = 'nord_rainbow'
    #cmap = plt.cm.YlGn
    if is_face:
        cs = ax.contourf(X, Y, Z, cmap = cmap, levels = levels)
    else:
        cs = ax.contour(X, Y, Z, cmap = cmap, levels = levels)
        ax.clabel(cs)
    if is_face : 
        fig.colorbar(cs)
    # ax.set_xlim(-0.02, 0.02)
    # ax.set_ylim(-0.02, 0.02)
    
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    ax.scatter(X[min_idx], Y[min_idx])
    ax.scatter(0.0, 0.0, marker = 'x')
    
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done.')
    

def make_3d_surface(X, Y, Z, figname):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    
    if False:
        ax.plot_surface(X, Y, Z, cmap = plt.cm.viridis)
    else:
        ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), cmap = 'nord_rainbow')
        
    ax.view_init(azim = 120, elev = 5)
    
    ax.set_zlim(0.0, 8.0)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(figname, dpi=300)
    plt.close()
    
    print(f'{figname} is done.')


def main(opts):
    fig_dir = f'figs/results/{opts.model}/{opts.classifier}/{opts.id}/{opts.seed}'
    
    mkdir(fig_dir)
    
    ver = opts.ver
    file = f'results/{opts.model}/{opts.classifier}/{opts.id}/{opts.seed}/loss_landscape_{ver}.npz'
    print(file)
    data = np.load(file, allow_pickle = True)
    x = data['xcoordinates']
    y = data['ycoordinates']
    loss = data['loss']
    
    X, Y = np.meshgrid(x, y)
    Z = np.array(loss[:])
    print(Z.shape)
    print(f'min : {np.min(Z)}')
    print(f'max : {np.max(Z)}')
    
    figname = f'{fig_dir}/plot_loss_landscape_{ver}'
    make_3d_surface(X, Y, Z, f'{figname}_surface.png')
    make_2d_contorf(X, Y, Z, n_level = 20, figname = f'{figname}_contor.png',  is_face = False, z_limit = None)
    make_2d_contorf(X, Y, Z, n_level = 20, figname = f'{figname}_contorf.png', is_face = True,  z_limit = None)
    
    

if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-s',  '--seed',       action = 'store', dest = 'seed',       type = int, default = 3407)
    parser.add_argument('-m',  '--model',      action = 'store', dest = 'model',      type = str, default = 'MLPMixer')
    parser.add_argument('-cl', '--classifier', action = 'store', dest = 'classifier', type = str, default = 'MLP')
    parser.add_argument('-i',  '--id',         action = 'store', dest = 'id',         type = str, default = None)
    parser.add_argument('-gi', '--gpu_id',     action = 'store', dest = 'gpu_id',    default = 4,  type = int)
    parser.add_argument('-nf', '--nfiles',     action = 'store', dest = 'n_files',   default = 8,  type = int)
    parser.add_argument('-ver', '--ver',       action = 'store', dest = 'ver',       default = 'ver1.train',  type = str)
    opts = parser.parse_args()
    
    
    main(opts)
