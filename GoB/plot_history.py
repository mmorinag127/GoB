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

colors = [nord_color.color(c) for c in ('red', 'green', 'blue', 'orange', 'violet', 'frost light blue',) ]
colors = [nord_color.color(c) for c in ('frost green', 'green', 'yellow', 'orange', 'red',
                                        'violet', 'frost light blue',  'blue', 'light0') ]

def plot_line(x, y, best_epoch, xlabel, lims, figname):
    plt.figure(figsize=(8.0, 6.0))
    fig, ax = plt.subplots()
    
    n_col = len(list(x['train'].keys()))
    for i, label in enumerate(x['train'].keys()):
        if n_col == 1:
            ax.plot(x['train'][label], y['train'][label], '-',  alpha = 1.0, label = f'train: {xlabel[0]}', color = colors[i])
            ax.plot(x['test'][label],  y['test'][label],  '--', alpha = 1.0, label = f'test: {xlabel[0]}',  color = colors[i])
        else:
            ax.plot(x['train'][label], y['train'][label], '-',  alpha = 1.0, label = f'{xlabel[0]}', color = colors[i])
            ax.plot(x['test'][label],  y['test'][label],  '--', alpha = 1.0,                         color = colors[i])
    
    
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
        leg = ax.legend(ncol = math.ceil(n_col / 2), loc='lower center', frameon=False)
    else:
        leg = ax.legend(frameon=False)
    
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    # print(f'{figname} is done...')

def plot_history(workdir, results, metrics = ['loss', 't1_acc', 't2_acc', 'time', 'GPU', 'g_norm', 'p_norm'], phases = ('test', 'train')):
    
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
    
    Xs, Ys = {}, {}
    
    for metric in metrics:
        
        if metric not in results['train']:
            print(metric, 'skip')
            continue
        
        X, Y = {}, {}
        for phase in phases:
            d = results[phase][metric]
            d = np.array(d)
            X[phase] = {}
            Y[phase] = {}
            
            for i, v in enumerate(keywords[metric]):
                if d.ndim > 1:
                    Y[phase][v] = d[:, i]
                else:
                    Y[phase][v] = d[:]
                X[phase][v] = epochs
        
        labels = np.array(keywords[metric])
        figname = f'{workdir}/history/plot_{metric}.png'
        plot_line(x = X, y = Y, best_epoch = best_epoch, xlabel = ['epoch', metric], lims = None, figname = figname)
        Xs[metric] = X
        Ys[metric] = Y
    return Xs, Ys

def main(opts):
    filename = f'{opts.workdir}/history-{opts.model_name}.json'
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
    metrics = ['loss', 't1_acc', 't2_acc', 'time', 'GPU', 'g_norm', 'p_norm']
    X, Y = plot_history(opts.workdir, data[0], metrics = metrics)
    return X, Y





if __name__ == '__main__':
    #from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-w',  '--workdir', action = 'store', dest = 'workdir',     type = str, default = '')
    parser.add_argument('-mn', '--model_name',   action = 'store', dest = 'model_name', type = str, default = 'nominal')
    parser.add_argument('-ws', '--workdirs', action = 'store', dest = 'workdirs',     type = str, default = '')
    opts = parser.parse_args()
    main(opts)
    
    
    
    
    
    
    
    
    
    