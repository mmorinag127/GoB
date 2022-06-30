import torch

import os

import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


from color import nord_color

colors = [nord_color.color(c) for c in ('red', 'green', 'blue', 'orange', 'violet', 'frost light blue',) ]
colors = [nord_color.color(c) for c in ('frost green', 'green', 'yellow', 'orange', 'red',
                                        'violet', 'frost light blue',  'blue', 'light0') ]

def test_lr():
    
    params = [torch.tensor(-75.0, requires_grad=True), torch.tensor(5.0, requires_grad=True)]
    optimizer = torch.optim.Adam(lr = 1.0e-3, weight_decay = 5.0e-4, params = params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer = optimizer )
    
    print(type(scheduler))
    if type(scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
        print('!!!!!!!!!!!!')

def plot_lr(figname):
    params = [torch.tensor(-75.0, requires_grad=True), torch.tensor(5.0, requires_grad=True)]

    optimizer = torch.optim.Adam(lr = 1.0e-3, weight_decay = 5.0e-4, params = params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts( optimizer = optimizer, T_0 = 20, T_mult = 2, eta_min = 1.0e-6)
    
    epochs = []
    lrs = []
    
    for epoch in range(100):
        optimizer.step()
        scheduler.step( epoch )
        lr = optimizer.param_groups[0]['lr']
        epochs.append( epoch )
        lrs.append(lr)
    
    epochs = np.array(epochs)
    lrs = np.array(lrs)
    
    plt.figure(figsize=(8.0, 6.0))
    fig, ax = plt.subplots()
    
    plt.plot(epochs, lrs, '-', alpha = 1.0)
    
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_line(x, y, best_epoch, xlabel, lims, figname):
    plt.figure(figsize=(8.0, 6.0))
    fig, ax = plt.subplots()
    
    n_col = len(list(x['train'].keys()))
    for i, label in enumerate(x['train'].keys()):
        if n_col == 1:
            ax.plot(x['train'][label], y['train'][label], '-',  alpha = 1.0, label = f'train: {label}', color = colors[i])
            ax.plot(x['test'][label],  y['test'][label],  '--', alpha = 1.0, label = f'test: {label}', color = colors[i])
        else:
            ax.plot(x['train'][label], y['train'][label], '-',  alpha = 1.0, label = f'{label}', color = colors[i])
            ax.plot(x['test'][label],  y['test'][label],  '--', alpha = 1.0,                     color = colors[i])
    
    
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
        import math
        leg = ax.legend(ncol = math.ceil(n_col / 2), loc='lower center', frameon=False)
    else:
        leg = ax.legend(frameon=False)
    
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_lines(X, Y, best_epoch, lims, figname ):
    fig = plt.figure(figsize=(8.0, 12.0 ) )
    gs = fig.add_gridspec(nrows = len(X.keys()), hspace=0)
    ax = gs.subplots(sharex=True, sharey=False)
    
    for i, key in enumerate(X.keys()):
        x, y = X[key], Y[key]
        for j, label in enumerate(x['train'].keys()):
            ax[i].plot(x['train'][label], y['train'][label], '-',  alpha = 1.0, label = f'train: {label}', color = colors[j])
            ax[i].plot(x['test'][label],  y['test'][label],  '--', alpha = 1.0, label = f'test: {label}', color = colors[j])
        
        ax[i].axvline(x = best_epoch, ymin = 0.025, ymax = 0.975, color = nord_color.color('dark2'), linestyle = '--')
        ax[i].set_xlabel('epoch', fontsize=20)
        ax[i].set_ylabel(key,     fontsize=20)
        ax[i].grid()
        ax[i].legend()
    
    #plt.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def make_label(label_map):
    
    labels = {}
    
    labels = { val:[] for val in list(set(label_map.values()))}
    for key, val in label_map.items():
        labels[val].append(key.replace('-jet', ''))
    
    ret = []
    for val in list(set(label_map.values())):
        ret.append('/'.join(f'{s}'for s in labels[val])+'-jet')
    print(ret)
    return ret

def read_result_json(filename):
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
    return data

def plot_results( model, classifier, id, seed, model_name, is_plot = True) : 
    #labels = { s:l for l, s in zip(config.dataset.args.labels, config.dataset.args.samples) }
    
    results = read_result_json(f'results/{model}/{classifier}/{id}/{seed}/history-{model_name}.json')[0]
    label_map = results['label_map']
    
    
    metrics = {'loss':['loss'], 
                'acc':['accuracy'], 
                'lr':['learning rate'], 
                'batch_size':['batch size'], 
                'process_per_sec':['#of process/sec'], 
                'accs': make_label(label_map)
                }
    
    phases = ('test', 'train')
    
    save_dir = f'figs/results/{model}/{classifier}/{id}/{seed}/{model_name}'
    mkdir(save_dir)
    
    best_epoch = results['train']['best_epoch'][-1][0]
    
    X, Y = {}, {}
    epochs = np.array(results['train']['epoch'])[:,0]
    for key, val in metrics.items():
        if key not in results['train']:
            continue
        X[key] = {}
        Y[key] = {}
        
        for phase in phases:
            d = results[phase][key]
            d = np.array(d)
            Y[key][phase] = {}
            X[key][phase] = {}
            
            for i, v in enumerate(val):
                Y[key][phase][v] = d[:,i]
                X[key][phase][v] = epochs
        
        labels = np.array(val)
        if is_plot:
            plot_line(x = X[key],  y = Y[key], best_epoch = best_epoch, xlabel = ['epoch', key], lims = None, figname = f'{save_dir}/history-{key}.png')
        
    if is_plot:
        plot_lines(X, Y, best_epoch = best_epoch, lims = None, figname = f'{save_dir}/history-all.png')
    
    return X, Y

def NiB(N, k):
    n = N / 1024
    if n < 512:
        return n, k+1
    
    return NiB(n, k+1)

def NiB_format(N):
    n, k = NiB(N, 0)
    iB = {0:'iB', 1:'kiB', 2:'MiB', 3:'GiB', 4:'PiB'}
    if n > 1.0:
        n = int(n)
    return f'{n}{iB[k]}'

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def comparison(opts):
    
    save_dir = f'figs/results/{opts.model}/{opts.classifier}/comparison/{opts.seed}'
    mkdir(save_dir)
    metrics = {'loss':['loss'], 'acc':['accuracy'], 
                #'lr':['learning rate'], 
                #'accs':['d-jet', 'u-jet', 's-jet', 'c-jet', 'b-jet', 'g-jet'] 
                #'accs':['l-jet', 'c-jet', 'b-jet', 'g-jet'] 
                }

    n_per_file = 65536
    is_plot = True
    n_files = [8, 16, 32, 64]
    n_channels = [32]
    n_depths = [2]
    
    X, Y = {}, {}
    for n_file in n_files:
        X[n_file] = {}
        Y[n_file] = {}
        for n_channel in n_channels:
            X[n_file][n_channel] = {}
            Y[n_file][n_channel] = {}
            for n_depth in n_depths:
                x, y = plot_results(model = opts.model, classifier = opts.classifier, id = f'{n_file}f/{n_channel}c/{n_depth}d', seed = opts.seed, is_plot = is_plot)
                X[n_file][n_channel][n_depth] = x
                Y[n_file][n_channel][n_depth] = y
    
    def make_n_file_comparison(key, val, n_channel, n_depth):
        import math
        for v in val:
            x = {'train':{}, 'test':{}}
            y = {'train':{}, 'test':{}}
            for n_file in n_files:
                n_sample = n_file * n_per_file
                label = NiB_format(n_sample)
                for phase in ['train', 'test']:
                    x[phase][label] = X[n_file][n_channel][n_depth][key][phase][v]
                    y[phase][label] = Y[n_file][n_channel][n_depth][key][phase][v]
                
            figname = f'{save_dir}/plot_n_file_comparison_{n_channel}-{n_depth}-{key}-{v}.png'
            
            plot_line(x = x, y = y, best_epoch = None, lims = None, xlabel = ['epoch', key], figname = figname )
            
            
    
    n_channel = 64
    n_depth = 2
    for key, val in metrics.items():
        make_n_file_comparison(key, val, n_channel = n_channel, n_depth = n_depth)
    
if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-m', '--model',      action = 'store', dest = 'model',      default = None,  type = str)
    parser.add_argument('-c', '--classifier', action = 'store', dest = 'classifier', default = None,  type = str)
    parser.add_argument('-i', '--id',         action = 'store', dest = 'id',         default = None,  type = str)
    parser.add_argument('-s', '--seed',       action = 'store', dest = 'seed',       default = 3407,  type = int)
    parser.add_argument('-mn','--model_name', action = 'store', dest = 'model_name', default = 'nominal',  type = str)
    opts = parser.parse_args()
    
    if opts.id is not None:
        plot_results(opts.model, opts.classifier, opts.id, opts.seed, opts.model_name)
    else:
        #plot_lr('figs/cos.1.png')
        comparison(opts)




