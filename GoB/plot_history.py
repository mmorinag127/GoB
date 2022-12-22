from copy import deepcopy
import math, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg') 

plt.rcParams.update({'figure.max_open_warning': 0})

# plt.style.use('/home/masahiro.morinaga/.matplotlib/stylelib/nord-light.mplstyle')
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys
sys.path.append('plotting/')
import argparse


from color import nord_color

colors0 = [nord_color.color(c) for c in ('red', 'green', 'blue', 'orange', 'violet', 'frost light blue',) ]
colors = [nord_color.color(c) for c in ('frost green', 'green', 'yellow', 'orange', 'red',
                                        'violet', 'frost light blue',  'blue', 'light0') ]

rainbow = plt.cm.get_cmap('nord_rainbow', 100)

def mkdir(dir):
    if not os.path.exists(f'{dir}'):
        os.makedirs(f'{dir}')

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def plot_line(x, y, best_epoch, xlabel, lims, figname, is_log = 'None', is_plot = 'test/train'):
    plt.figure(figsize=(8.0, 6.0))
    fig, ax = plt.subplots()
    
    
    

    if type(x['train']) is dict:
        n_col = len(list(x['train'].keys()))
        rcolors = rainbow(np.linspace(0, 1.0, n_col))
        for i, label in enumerate(x['train'].keys()):
            if 'train' in is_plot:
                ax.plot(x['train'][label], y['train'][label], '-', alpha = 1.0, label = label, color = rcolors[i])
            
            elif 'test' in is_plot:
                ax.plot(x['test'][label],  y['test'][label],  '-', alpha = 1.0, label = label, color = rcolors[i])
    else:
        n_col = 1
        ax.plot(x['train'], y['train'], '-', alpha = 1.0, label = f'train', color = colors[7])
        ax.plot(x['test'],  y['test'],  '-', alpha = 1.0, label = f'test',  color = colors[4])
        
    if best_epoch is not None:
        ax.axvline(x = best_epoch, ymin = 0.025, ymax = 0.975, #ymin = np.min(y), ymax = np.max(y), 
                    color = nord_color.color('dark2'), linestyle = '--')#, label = 'best point')
    
    if lims is not None : 
        plt.xlim(*lims)
        plt.ylim(*lims)
        
    if xlabel:
        ax.set_xlabel(xlabel[0], fontsize=20)
        ax.set_ylabel(xlabel[1], fontsize=20)
    
    ax.xaxis.set_tick_params(labelsize=16)#,bottom=True,top=True)
    ax.yaxis.set_tick_params(labelsize=16)#,left=True,right=True)
    ax.grid()
    
    if n_col > 200 : 
        
        ax.plot(np.NaN, np.NaN, '-',  alpha = 1.0, label = 'train', color = nord_color.color('dark0'))
        ax.plot(np.NaN, np.NaN, '--', alpha = 1.0, label = 'valid', color = nord_color.color('dark0'))
        #ax.get_yaxis().set_visible(False)
        #ax.legend(loc='upper left')
    if n_col > 4: 
        leg = ax.legend(ncol = math.ceil(n_col / 2), loc='upper center', frameon=False)
    else:
        leg = ax.legend(frameon=False)
    
    if 'x' in is_log:
        plt.xscale('log')
    if 'y' in is_log:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    # print(f'{figname} is done...')

def plot_history(workdir, results, metrics = ['loss', 't1_acc', 't2_acc', 'time', 'GPU', 'g_norm', 'p_norm'], phases = ('test', 'train'), burn_in = -1):
    
    keywords = {
        'loss': ['loss'],
        't1_acc': ['top1 accuracy'],
        't2_acc': ['top2 accuracy'],
        't5_acc': ['top5 accuracy'],
        'lr':['learning rate'], 
        'batch_size':['batch size'], 
        'process_per_sec':['image/sec'], 
        'time':['training time [sec]'],
        'GPU': ['GPU usage(all mean) [%]'],
        'g_norm': ['Global Gradient Norm'],
        'p_norm': ['Parameters Norm'],
    }
    keywords.update({f'GPU{i}':[f'GPU{i} usage'] for i in range(8)})
    
    best_epoch = results['best_epoch']
    epochs = np.array(results['epoch'])

    if not os.path.exists(f'{workdir}/history'):
        os.makedirs(f'{workdir}/history')
    
    Xs, Ys, YYs = {}, {}, {}
    
    for metric in metrics:
        
        if metric not in results['train']:
            print(metric, 'skip')
            continue
        
        X, Y, YY = {}, {}, {}
        for phase in phases:
            d = results[phase][metric]
            d = np.array(d)
            X[phase] = {}
            Y[phase] = {}
            
            if len(keywords[metric]) > 1:
                for i, v in enumerate(keywords[metric]):
                    Y[phase][v] = d[:, i]
                    X[phase][v] = epochs
                    
                    if burn_in > 0:
                        X[phase][v] = X[phase][v][burn_in:]
                        Y[phase][v] = Y[phase][v][burn_in:]
            else:
                Y[phase] = d[:]
                X[phase] = epochs
                
                if burn_in > 0:
                    X[phase] = X[phase][burn_in:]
                    Y[phase] = Y[phase][burn_in:]
                
        
        is_log =  epochs[-1] > 100
        is_log = ''
        labels = np.array(keywords[metric])
        figname = f'{workdir}/history/plot_{metric}.png'
        
        plot_line(x = X, y = Y, best_epoch = best_epoch, xlabel = ['epoch', keywords[metric][0]], lims = None, figname = figname, is_log = is_log)
        Xs[metric] = X
        Ys[metric] = Y
        
        # for phase in phases:
        #     yy = np.insert(Y[phase][metric], 0, Y[phase][metric][0])
        #     yy = np.delete(yy, -1)
        #     yy = 1.0 - yy/Y[phase][metric]
        #     YY[phase] = {metric:yy}
        # figname = f'{workdir}/history/plot_{metric}-diff.png'
        # plot_line(x = X, y = YY, best_epoch = best_epoch, xlabel = ['epoch', metric+'(diff=1-y0/y1)'], lims = None, figname = figname, is_log = is_log)
        
        
    
    return Xs, Ys

def plot_all_history(workdir, model_name, burn_in, metrics):
    filename = f'{workdir}/history-{model_name}.json'
    import glob
    import json
    files = glob.glob(filename)
    
    print(filename)
    if len(files) == 0 :
        raise ValueError(f'There is no files : {filename}')
    data = []
    for filename in files:
        if filename.startswith("_"):
            continue
        with open(filename, 'r') as f:
            data.extend(json.load(f))
    
    
    #print(data)
    
    X, Y = plot_history(workdir[0], data[0], metrics = metrics, burn_in = burn_in)
    return X, Y

def main(opts):
    metrics = ['loss', 't1_acc', 't2_acc', 'time', 'GPU', 'g_norm', 'p_norm']
    
    if 'nominal' in opts.mode:
        plot_all_history(opts.workdir, opts.model_name, opts.burn_in, metrics)
        
    elif 'comp' in opts.mode:
        X, Y = {}, {}
        for metric in metrics:
            X[metric] = {t:{} for t in ['train', 'test']}
            Y[metric] = {t:{} for t in ['train', 'test']}
        
        workdirs = opts.workdir
        workdirs.sort(key=natural_keys, reverse=True)
        for workdir in workdirs:
            key = workdir.split('/')[-3]
            key = '-'.join(k for k in key.split('-')[:3])
            
            x, y = plot_all_history(workdir, opts.model_name, opts.burn_in, metrics)
            for metric in metrics:
                for t in ['train', 'test']:
                    
                    X[metric][t][key] = x[metric][t]
                    Y[metric][t][key] = y[metric][t]
        
        mkdir(opts.comp_dir)
        for metric in metrics:
            figname = f'{opts.comp_dir}/plot_{metric}-train.png'
            plot_line(x = X[metric], y = Y[metric], best_epoch=None, xlabel=['epoch', metric], lims = None, figname = figname, is_plot='train', is_log = '')
            
            figname = f'{opts.comp_dir}/plot_{metric}-test.png'
            plot_line(x = X[metric], y = Y[metric], best_epoch=None, xlabel=['epoch', metric], lims = None, figname = figname, is_plot='test', is_log = '')
        





if __name__ == '__main__':
    #from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-w',  '--workdir', action = 'store', dest = 'workdir',     type = str, nargs = '*')
    parser.add_argument('-mn', '--model_name',   action = 'store', dest = 'model_name', type = str, default = 'nominal')
    parser.add_argument('-m',  '--mode',     action = 'store', dest = 'mode',         type = str, default = 'nominal')
    parser.add_argument('-cd', '--comp_dir', action = 'store', dest = 'comp_dir',     type = str, default = '')
    parser.add_argument('-bi', '--burn_in',  action = 'store', dest = 'burn_in',      type = int, default = -1)
    opts = parser.parse_args()
    main(opts)
    
    
    
    
    
    
    
    
    
    