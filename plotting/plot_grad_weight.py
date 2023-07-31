from copy import deepcopy
import math, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg') 

from matplotlib.backends.backend_pdf import PdfPages

# plt.style.use('/home/masahiro.morinaga/.matplotlib/stylelib/nord-light.mplstyle')
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys
import argparse
import re

from color import nord_color

colors0 = [nord_color.color(c) for c in ('red', 'green', 'blue', 'orange', 'violet', 'frost light blue',) ]
colors = [nord_color.color(c) for c in ('frost green', 'green', 'yellow', 'orange', 'red',
                                        'violet', 'frost light blue',  'blue', 'light0') ]

rainbow = plt.cm.get_cmap('nord_rainbow', 100)

def mkdir(dir):
    if not os.path.exists(f'{dir}'):
        os.makedirs(f'{dir}')

def plot_lines(x, y, title, pdf):
    plt.figure(figsize=(8.0, 6.0))
    fig, ax = plt.subplots()
    
    n_param = y.shape[1]
    colors = nord_color.get_colors(n_param, name = 'nord_rainbow')
    
    for i in range(n_param):
        # if np.isnan(y[:,i]).any():
        #     print('NaN!!', title)
        ax.plot(x, y[:,i], '-', alpha = 0.25, color = colors[i], linewidth=1)
    
    ax.xaxis.set_tick_params(labelsize=16)#,bottom=True,top=True)
    ax.yaxis.set_tick_params(labelsize=16)#,left=True,right=True)
    ax.grid()
    
    plt.title(title, loc='left')
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    # print(f'{figname} is done...')

def make_weight_history(wdir, figname):
    import glob
    files = glob.glob(f'{wdir}/0*.npz')
    
    w = np.load(files[0], allow_pickle=True)
    weights = {}
    for key, val in w.items():
        weights[key] = [val]
    
    for i,f in enumerate(files[1:]):
        w = np.load(f, allow_pickle=True)
        total = 0.0
        for key, val in w.items():
            weights[key].append(val)
            total += np.sum(val**2)
        total = total**0.5
        print(f'{i} : {total}')
    
    with PdfPages(figname) as pdf:
        print(f'{figname} is making')
        x = np.arange(len(files))
        
        for key, val in weights.items():
            y = np.stack(val, axis=0)
            if y.ndim > 2:
                n = y.shape[0]
                y = np.reshape(y, [n, -1])
            
            plot_lines(x, y, key, pdf)

def main(opts):
    
    # gradient 
    figdir = f'{opts.workdir}/figs/grad/'
    mkdir(figdir)
    make_weight_history(f'{opts.workdir}/grad',   figname = f'{figdir}/grad.pdf')
    make_weight_history(f'{opts.workdir}/param', figname = f'{figdir}/param.pdf')





if __name__ == '__main__':
    #from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-w',  '--workdir', action = 'store', dest = 'workdir',     type = str, default=None)
    opts = parser.parse_args()
    main(opts)
    
    
    
    
    
    
    
    
    
    