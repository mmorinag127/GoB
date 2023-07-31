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
import re

from color import nord_color

colors0 = [nord_color.color(c) for c in ('red', 'green', 'blue', 'orange', 'violet', 'frost light blue',) ]
colors = [nord_color.color(c) for c in ('frost green', 'green', 'yellow', 'orange', 'red',
                                        'violet', 'frost light blue',  'blue', 'light0') ]
colors3 = [nord_color.color(c) for c in ['red', 'blue', 'green']*10]


rainbow = plt.cm.get_cmap('nord_rainbow', 100)

def mkdir(dir):
    if not os.path.exists(f'{dir}'):
        os.makedirs(f'{dir}')

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

def plot_simple(x, y, axis_names, figname, is_log=''):
    plt.figure(figsize=(8.0, 6.0))
    fig, ax = plt.subplots()
    
    if type(y) is dict:
        for i,(k,v) in enumerate(y.items()):
            ax.plot(x, v, '-', alpha = 1.0, color = colors3[i], label=k)
    else:
        ax.plot(x, y, '-', alpha = 1.0, color = colors3[0])
    
    ax.set_xlabel(axis_names[0], fontsize=20)
    ax.set_ylabel(axis_names[1], fontsize=20)
    if type(x) is dict:
        leg = ax.legend(frameon=False)
    ax.xaxis.set_tick_params(labelsize=16)#,bottom=True,top=True)
    ax.yaxis.set_tick_params(labelsize=16)#,left=True,right=True)
    plt.tick_params(axis='x', which='both')
    if 'x' in is_log:
        plt.xscale('log')
    if 'y' in is_log:
        plt.yscale('log')
    
    ax.grid(which='both')
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    # print(f'{figname} is done...')

def plot_history(fig_dir, results, metrics = ['loss', 't1_acc', 't2_acc', 'time', 'GPU', 'g_norm', 'p_norm'], phases = ('test', 'train'), burn_in = -1):
    
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
        'load_loss': ['load loss'],
        'imp_loss': ['Importance loss'],
        'expert_label_loss':['label importance loss'], 
    }
    keywords.update({f'GPU{i}':[f'GPU{i} usage'] for i in range(8)})
    
    best_epoch = results['best_epoch']
    epochs = np.array(results['epoch'])

    mkdir(f'{fig_dir}')
    
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
        figname = f'{fig_dir}/plot_{metric}.png'
        
        plot_line(x = X, y = Y, best_epoch = best_epoch, xlabel = ['step', keywords[metric][0]], lims = None, figname = figname, is_log = is_log)
        Xs[metric] = X
        Ys[metric] = Y
    
    return Xs, Ys

def plot_all_history(workdir, model_name, burn_in, metrics, comp_dir = None):
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
    fig_dir = f'{workdir}/figs/history'
    if comp_dir is not None:
        fig_dir = comp_dir
    X, Y = plot_history(fig_dir, data[0], metrics = metrics, burn_in = burn_in)
    print(fig_dir)
    return X, Y

def copy_figs_from_result(workdir,  comp_dir):
    if not os.path.exists(f'{workdir}'):
        print(f'workdir: {workdir} is not exists, just skip...')
        return
    import glob
    from distutils.dir_util import copy_tree
    for d in glob.glob(f'{workdir}/figs/*'):
        copy_tree(d, f'{comp_dir}/')

def plot_weight_history(x, y, bins, axis_names, figname):
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    
    gs = fig.add_gridspec(ncols = 1, nrows = 1, hspace=0.1, wspace=0.05)
    ax = gs.subplots(sharex=True, sharey=True)
    
    cmap = plt.cm.get_cmap('nord_mono_red', 100)
    
    h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=None, rasterized=True)
    #fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
    ax.set_xlabel(axis_names[0], fontsize=16)
    ax.set_ylabel(axis_names[1], fontsize=16)
    ax.grid()
    # ax.legend()
    ax.set_title(axis_names[2])
    ax.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    #print(f'{figname} is done...')

def plot_weight(steps, weight, wtype, wname, figname):
    N = weight.shape[0]
    y = np.reshape(weight, -1)
    M = y.shape[0]//N
    x = np.repeat(steps, M, axis=-1)
    bins = [N, 200]
    plot_weight_history(x, y, bins=bins, axis_names = ['steps/1000', wtype, wname], figname = figname+'-hist.png')
    
    y = np.reshape(weight, (N, -1))
    norms = np.linalg.norm(y, ord=2, axis= -1)
    plot_simple(x = steps, y = norms, axis_names=['steps/1000', 'norm', wname], figname=figname+'-norm.png')

def plot_all_weight_history(workdir):
    fig_dir = f'{workdir}/figs/weights'
    
    for wtype in ['param', 'grad']:
        mkdir(f'{fig_dir}/{wtype}/')
        weights = np.load(f'{workdir}/{wtype}.npz')
        steps = weights['steps']/1000
        for key, weight in weights.items():
            
            if key == 'steps':
                continue
            wname = key.replace('model/', '')
            wname = wname.replace('/', '-')
            if 'lift' in key:
                for i in range(weight.shape[1]):
                    w = weight[:, i, ...]
                    figname = f'{fig_dir}/{wtype}/{wname}-expert{i}'
                    plot_weight(steps, w, wtype, f'{wname}-expert{i}', figname)
            else:
                figname = f'{fig_dir}/{wtype}/{wname}'
                plot_weight(steps, weight, wtype, wname, figname)

def plot_lr_range_test(workdir, model_name='nominal'):
    def clip_loss(loss, c=1.0):
        m = loss > c
        loss[m] = c
        return loss
    fig_dir = f'{workdir}/figs/lr-test'
    mkdir(fig_dir)
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
    results = data[0]
    lr = np.array(results['train']['lr'])
    loss_train = clip_loss(np.array(results['train']['loss']))
    loss_test = clip_loss(np.array(results['test']['loss']))
    
    steps = np.array(results['epoch'])
    plot_simple(steps, loss_train, ['step', 'train loss'], f'{fig_dir}/loss_train.png')
    plot_simple(steps, loss_test,  ['step', 'test loss'],  f'{fig_dir}/loss_test.png')
    plot_simple(steps, {'train':loss_train, 'test':loss_test},  ['step', 'loss'],  f'{fig_dir}/loss.png')
    plot_simple(steps, lr,   ['step', 'learning rate'], f'{fig_dir}/lr.png')
    
    plot_simple(lr, loss_train, ['learning rate', 'loss train'], f'{fig_dir}/lr-loss_train.png', is_log='x')
    plot_simple(lr, loss_test,  ['learning rate', 'loss test'],  f'{fig_dir}/lr-loss_test.png', is_log='x')
    plot_simple(lr, {'train':loss_train, 'test':loss_test},  ['learning rate', 'loss'],  f'{fig_dir}/lr-loss.png', is_log='x')
    dlr = lr[1:] - lr[:-1]
    dloss_train = loss_train[1:] - loss_train[:-1]
    dloss_test = loss_test[1:] - loss_test[:-1]
    plot_simple(dlr, dloss_train, ['delta learning rate', 'delta loss(train)'], f'{fig_dir}/dlr-dloss_train.png', is_log='x')
    plot_simple(dlr, dloss_test, ['delta learning rate', 'delta loss(test)'], f'{fig_dir}/dlr-dloss_test.png', is_log='x')
    plot_simple(dlr, {'train':dloss_train, 'test':dloss_test}, ['delta learning rate', 'delta loss'], f'{fig_dir}/dlr-dloss.png', is_log='x')
    
    #print(data)
    

def main(opts):
    metrics = ['loss', 't1_acc', 't2_acc', 'g_norm', 'p_norm', 'lr']
    if opts.MoE:
        metrics += ['load_loss', 'imp_loss', 'expert_label_loss']
    
    if 'nominal' in opts.mode:
        plot_all_history(opts.workdir[0], opts.model_name, opts.burn_in, metrics)
    elif 'weight' in opts.mode:
        plot_all_weight_history(opts.workdir[0])
    elif opts.mode in ['lr-test', 'lr-range-test']:
        plot_lr_range_test(opts.workdir[0])
        
    elif 'comp' in opts.mode or opts.comp_dir is not None:
        mkdir(opts.comp_dir)
        X, Y = {}, {}
        for metric in metrics:
            X[metric] = {t:{} for t in ['train', 'test']}
            Y[metric] = {t:{} for t in ['train', 'test']}
        
        workdirs = opts.workdir
        workdirs.sort(key=natural_keys, reverse=True)
        print(opts.workdir)
        print(workdirs)
        keys = []
        for workdir in workdirs:
            key = workdir.split('/')[-2]
            #key = key.split('-')
            key = key[9:-5]
            print(key)
            keys.append(set(key))
        
        comm = keys[0]
        for key in keys:
            comm = comm & key
        
        for workdir in workdirs:
            key = workdir.split('/')[-2]
            #key = key.split('-')
            key = key[9:-5]
            #key = set(key) ^ comm
            key = '-'.join(k for k in sorted(key))
            print('key', key)
            
            x, y = plot_all_history(workdir, opts.model_name, opts.burn_in, metrics, comp_dir=f'{opts.comp_dir}/{key}')
            copy_figs_from_result(workdir, comp_dir=f'{opts.comp_dir}/{key}')
            for metric in metrics:
                for t in ['train', 'test']:
                    
                    X[metric][t][key] = x[metric][t]
                    Y[metric][t][key] = y[metric][t]
        
        mkdir(opts.comp_dir)
        for metric in metrics:
            print(metric)
            figname = f'{opts.comp_dir}/plot_{metric}-train.png'
            plot_line(x = X[metric], y = Y[metric], best_epoch=None, xlabel=['step', metric], lims = None, figname = figname, is_plot='train', is_log = '')
            
            figname = f'{opts.comp_dir}/plot_{metric}-test.png'
            plot_line(x = X[metric], y = Y[metric], best_epoch=None, xlabel=['step', metric], lims = None, figname = figname, is_plot='test', is_log = '')
        





if __name__ == '__main__':
    #from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-mn', '--model_name',   action = 'store', dest = 'model_name', type = str, default = 'nominal')
    parser.add_argument('-w',  '--workdir', action = 'store', dest = 'workdir',     type = str, nargs = '*')
    parser.add_argument('-m',  '--mode',     action = 'store', dest = 'mode',         type = str, default = 'nominal')
    parser.add_argument('-cd', '--comp_dir', action = 'store', dest = 'comp_dir',     type = str, default = None)
    parser.add_argument('-bi', '--burn_in',  action = 'store', dest = 'burn_in',      type = int, default = -1)
    parser.add_argument('-moe', '--moe',     action = 'store_true', dest = 'MoE')
    opts = parser.parse_args()
    main(opts)
    
    
    
    
    
    
    
    
    
    