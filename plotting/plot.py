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

import argparse
import re

from color import nord_color
def make_colors(n, N):
    rainbow = plt.cm.get_cmap('nord_rainbow', 256)
    a = np.linspace(256, 0, N)[n]
    a = np.linspace(256, 0, N)
    a = np.linspace(1, 0, N)
    r = rainbow(a)
    return r[n]

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

def get_color_line(i, j, I, J):
    if I == 2 and J == 1:
        c = 'blue' if i == 0 else 'red'
        return  nord_color.color(c), '-'
    if I == 2:# phase=2, 0 = train, 1 = test
        l = '-' if i == 0 else '--'
        c = make_colors(j, J)
        return c, l
    if J == 2:
        c = 'blue' if i == 0 else 'red'
        return  nord_color.color(c), '-'
    N = I*J
    n = i*I + j*J
    return make_colors(n, N), '-'

def plot_line(metric, best_epoch, figname, is_log = 'None', lims = None):
    plt.figure(figsize=(8.0, 6.0))
    fig, ax = plt.subplots()
    
    for x, y, label, color, line in metric.make_plot():
        if 'fill' in y['type']:
            ax.fill_between(x, y['up'], y['down'], interpolate=True, alpha = 0.5, label = label, facecolor = color)
        if 'line' in y['type']:
            ax.plot(x, y['y'], line, alpha = 1.0, label = label, color = color)
            
    
    if best_epoch is not None:
        ax.axvline(x = best_epoch, ymin = 0.025, ymax = 0.975, color = nord_color.color('dark2'), linestyle = '--')#, label = 'best point')
    
    if lims is not None:
        plt.xlim(*lims)
        plt.ylim(*lims)
        
    ax.set_xlabel(metric.get_title('x'), fontsize=20)
    ax.set_ylabel(metric.get_title('y'), fontsize=20)
    ax.xaxis.set_tick_params(labelsize=12)#,bottom=True,top=True)
    ax.yaxis.set_tick_params(labelsize=16)#,left=True,right=True)
    ax.grid()
    
    if metric.N > 1:
        for phase in metric.phases:
            ax.plot(np.NaN, np.NaN, '--', alpha = 1.0, label=phase, color = nord_color.color('dark0'))
    
    leg = ax.legend(frameon=False)
    
    if 'x' in is_log:
        plt.xscale('log')
    if 'y' in is_log:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_simple(p, title, figname, is_log=''):
    plt.figure(figsize=(8.0, 6.0))
    fig, ax = plt.subplots()
    
    for i,(k,xy) in enumerate(p.items()):
        x, y = xy
        c = make_colors(i, len(p))
        ax.plot(x, y, '-', alpha=1.0, label=k, color=c)
    
    ax.set_xlabel(title[0], fontsize=20)
    ax.set_ylabel(title[1], fontsize=20)
    ax.legend(frameon=False)
    ax.xaxis.set_tick_params(labelsize=16)#,bottom=True,top=True)
    ax.yaxis.set_tick_params(labelsize=16)#,left=True,right=True)
    plt.tick_params(axis='x', which='both')
    if 'x' in is_log:
        plt.xscale('log')
    if 'y' in is_log:
        plt.tick_params(axis='y', which='both')
        plt.yscale('log')
    
    ax.grid(which='both')
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    # print(f'{figname} is done...')

def plot_hist(x, title, figname, bins, brange, is_log='', norm=False):
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    gs = fig.add_gridspec(nrows = 1, ncols = 1, hspace=0)
    ax = gs.subplots(sharex=False, sharey=False)
    
    for i, (key, val) in enumerate(x.items()):
        weights = np.ones(len(val))/float(len(val)) if norm else None,
        c = make_colors(i, len(x))
        ax.hist(val, bins=bins, label=key, range=brange, weights = weights, alpha = 1.0, histtype = 'step', facecolor = c, zorder=0, edgecolor = c, linewidth = 1.5)
    if is_log:
        ax.set_yscale('log')
            
    ax.set_xlabel(title[0], fontsize=20)
    ax.set_ylabel(title[1], fontsize=20)
    ax.legend(frameon=False)
    ax.xaxis.set_tick_params(labelsize=16)#,bottom=True,top=True)
    ax.yaxis.set_tick_params(labelsize=16)#,left=True,right=True)
    plt.tick_params(axis='x', which='both')
    
    ax.set_xlim(*brange)
    plt.xlim(*brange)
    if 'x' in is_log:
        plt.xscale('log')
    if 'y' in is_log:
        plt.yscale('log')
    
    ax.grid(which='both')
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()

def plot_confution_matrix(_cm, labels, figname, norm = -1 ):
    fig = plt.figure(figsize=(6.0*1.06, 6.0) )
    gs = fig.add_gridspec(nrows = 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    
    cm = _cm.astype('float') / _cm.sum(axis=norm, keepdims = True) * 100.0 if norm >= 0 else _cm
    im = ax.imshow(cm, interpolation='nearest', cmap = 'nord_mono_blue')
    
    # We want to show all ticks...
    ax.set(xticks = np.arange(cm.shape[1]), yticks = np.arange(cm.shape[0]), xticklabels = labels, yticklabels = labels,
            ylabel = 'True label', xlabel = 'Predicted label', title = 'Efficiency' if norm == 1 else 'Purity' if norm == 0 else None)
    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label',      fontsize=16)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2. if norm < 0 else 50.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.1f}%' if norm >= 0 else f'{cm[i,j]}', ha = "center", va = "center", color = "white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.xlim(-0.5, len(labels)-0.5)
    plt.ylim(len(labels)-0.5, -0.5)
    
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

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

def plot_lr_range_test(workdir, _metrics, model_name='nominal', burn_in=0):
    fig_dir = f'{workdir}/figs/lr-test'
    mkdir(fig_dir)
    phases = ['train', 'test']
    filename = f'{workdir}/history-{model_name}.json'
    history = History(filename, {k:var_book(k, phases, burn_in) for k in _metrics})
    history.make_plots(fig_dir)
    c=0.01
    steps, lr = history.metrics['lr'].get_value('train')
    
    loss_train = np.clip(history.metrics['loss'].get_value('train', ret=False), a_min=-c, a_max=1.0)
    loss_test  = np.clip(history.metrics['loss'].get_value('test',  ret=False), a_min=-c, a_max=1.0)
    plot_simple({'train':[steps, loss_train], 'test':[steps, loss_test]}, title=['steps','loss'],          figname=f'{fig_dir}/loss-steps.png', is_log='')
    plot_simple({'train':[lr, loss_train],    'test':[lr, loss_test]},    title=['learning rate', 'loss'], figname=f'{fig_dir}/loss-lr.png',    is_log='x')
    plot_simple({'train':[steps, lr]},    title=['steps', 'learning rate'], figname=f'{fig_dir}/steps-lr.png',    is_log='y')
    dlr = lr[:-1]
    dloss_train = loss_train[1:] - loss_train[:-1]
    dloss_test = loss_test[1:] - loss_test[:-1]
    
    dloss_train = np.clip(dloss_train, a_min=-c, a_max=c)
    dloss_test = np.clip(dloss_test, a_min=-c, a_max=c)
    plot_simple({'train':[dlr, dloss_train], 'test':[dlr, dloss_test]}, title=['lr','delta loss'],          figname=f'{fig_dir}/dloss-dlr.png', is_log='x')

import json
class Variable:
    def __init__(self, key, title, name, phases, burn_in = 0, plot_type='line'):
        self.key = key
        self.name = name
        self.phases = phases
        self.N = len(self.name)
        self.value = {p:{} for p in self.phases}
        self.steps = None
        self.burn_in = burn_in if burn_in > 0 else 0
        self.title = title
        self.plot_type = plot_type
        self.magic_val = -999.
    
    def get_title(self, xy):
        if 'x' in xy:
            return self.xtitle
        if 'y' in xy:
            return self.title
    def titles(self):
        return [self.xtitle, self.title]
    def set(self, value, steps, xtitle):
        self.xtitle = xtitle
        self.steps = np.array(steps[self.burn_in:])
        
        for phase in self.phases:
            a = np.array(value[phase][self.key])
            for i, name in enumerate(self.name):
                self.value[phase][name] = a[self.burn_in:, i]
    
    def get_value(self, phase, idx=0, ret=True):
        if ret:
            return self.steps, self.value[phase][self.name[idx]]
        return self.value[phase][self.name[idx]]
    
    def make_plot(self):
        I = len(self.phases)
        J = self.N
        for i, phase in enumerate(self.phases):
            for j, (key, val) in enumerate(self.value[phase].items()):
                color, line = get_color_line(i, j, I, J)
                label = None
                if self.N == 1:
                    label = phase
                if self.N > 1:
                    if phase == 'train':
                        label = key
                
                if self.plot_type == 'line':
                    y = {'type':'line', 'y':val}
                elif self.plot_type == 'lines':
                    y = {'type':'line', 'y':val}
                    
                    
                elif self.plot_type == 'merge':
                    mean = np.mean(val, axis=-1, where=val!=self.magic_val)
                    down = np.min(val, axis=-1)
                    up   = np.max(val, axis=-1)
                    down = np.where(down!=self.magic_val, down, mean)
                    up   = np.where(up  !=self.magic_val, up, mean)
                    y = {'type':'fill_line', 'y':mean, 'up':up, 'down':down}
                yield self.steps, y, label, color, line
    
    def merge(self, others):
        
        idx = np.argmax(other.steps.shape[0]  for other in others)
        self.steps = others[idx].steps
        self.xtitle = others[idx].xtitle
        N = self.steps.shape[0]
        
        for phase in self.phases:
            self.value[phase] = {}
            for i, name in enumerate(self.name):
                A = []
                for other in others:
                    a = other.value[phase][name]
                    n = a.shape[0]
                    if n < N:
                        a_shape = list(a.shape)
                        a_shape[0] = N - n
                        a = np.concatenate([a, np.ones(a_shape)*self.magic_val], axis=0)
                    A.append(a)
                self.value[phase][name] = np.stack(A, axis=-1)

class History:
    def __init__(self, filename, metrics):
        self.filename = filename
        self.metrics = metrics
        if self.filename is not None:
            self.set()
    
    def set(self):
        data = []
        with open(self.filename, 'r') as f:
            data.extend(json.load(f))
        data = data[0]
        self.best_epoch = data['best_epoch']
        for metric in self.metrics.values():
            metric.set(data, data['epoch'], 'steps')
    
    def make_plots(self, figdir, is_log=''):
        for metric in self.metrics.values():
            figname=f'{figdir}/plot_{metric.key}.png'
            plot_line(metric, self.best_epoch, figname=figname, is_log=is_log)
    
    def merge(self, others):
        for key, metric in self.metrics.items():
            metric.merge([other.metrics[key] for other in others])
    
    def make_merge_plots(self, figdir, is_log=''):
        for metric in self.metrics.values():
            figname=f'{figdir}/plot_{metric.key}.png'
            plot_line(metric, None, figname=figname, is_log=is_log)

def var_book(metric, phases, burn_in, plot_type='line'):
    book = {
        'loss': ('loss', 'Loss', ['loss'], phases, burn_in), 
        't1_acc':('t1_acc', 'Top1 Accuracy', ['top1 accuracy'], ['test',], burn_in), 
        't2_acc':('t2_acc', 'Top2 Accuracy', ['top2 accuracy'], ['test',], burn_in), 
        'g_norm':('g_norm', 'Gradient Norm', ['g_norm'], ['train'], burn_in), 
        'p_norm':('p_norm', 'Parameter Norm', ['p_norm'], ['train'], burn_in), 
        'lr':('lr', 'Learning Rate', ['lr'], ['train'], burn_in), 
    }
    if metric not in book:
        raise ValueError(f'{metric} is not in book')
    return Variable(*book[metric], plot_type=plot_type)

def plot_all_history(workdir, model_name, burn_in, _metrics):
    filename = f'{workdir}/history-{model_name}.json'
    print(filename)
    phases = ['train', 'test']
    fig_dir = f'{workdir}/figs/history'
    mkdir(fig_dir)
    history = History(filename, {k:var_book(k, phases, burn_in) for k in _metrics})
    history.make_plots(fig_dir)
    return history

def plot_multi_history(comp_dir, workdirs, model_name, burn_in, _metrics):
    history = History(None, {k:var_book(k, ['train', 'test'], burn_in, 'merge') for k in _metrics})
    histories = []
    for workdir in workdirs:
        histories.append(plot_all_history(workdir, model_name, burn_in, _metrics))
    history.merge(histories)
    history.make_merge_plots(comp_dir)

def softmax(x):
    max_ = np.max(x, axis=1, keepdims=True)
    e_x  = np.exp(x - max_)
    sum_ = np.sum(e_x, axis=1, keepdims=True)
    f_x  = e_x / sum_
    return f_x
def transform_score(x):
    xmin, xmax = np.min(x), np.max(x)
    return (x - xmin)/(xmax - xmin)

def make_labels(table, name):
    idx = {k:i for i,k in enumerate(table['names'])}[name]
    labels = {}
    for key, val in table['samples'].items():
        if val not in labels.keys():
            labels[val] = []
        labels[val].append(key)

def make_roc_curve(scores, labels, xlabels):
    from sklearn import metrics
    one_hot = np.eye(len(xlabels))[labels]
    ret = {}
    for i in range(len(xlabels)):
        fpr, tpr, _ = metrics.roc_curve(one_hot[:, i], scores[:, i])
        auc = metrics.auc(fpr, tpr)
        ret[xlabels[i]] = {'fpr':fpr, 'tpr':tpr, 'auc':auc}
    return ret

def make_results(workdir, model_name, xlabels = None):
    from sklearn import metrics
    fig_dir = f'{workdir}/figs'
    
    data = np.load(f'{workdir}/eval-outputs.npz', allow_pickle=True)
    logits, labels = data['logit'], data['label'].astype(int)
    if xlabels is None:
        xlabels = make_labels(data['label_table'], data['label_name'])
        
    scores = transform_score(softmax(logits))
    # roc curve
    _roc = make_roc_curve(scores, labels, xlabels)
    roc1, roc2 = {}, {}
    for xlabel in xlabels:
        auc = _roc[xlabel]['auc']
        l = f'{xlabel}:{auc:.3f}'
        roc1[l] = [_roc[xlabel]['tpr'], _roc[xlabel]['fpr']]
        roc2[l] = [_roc[xlabel]['tpr'], 1./_roc[xlabel]['fpr']]
    
    plot_simple(roc1, title=['True positive', 'False positive'], figname=f'{fig_dir}/ROC1.png')
    plot_simple(roc1, title=['True positive', 'False positive'], figname=f'{fig_dir}/ROC2.png', is_log='y')
    plot_simple(roc2, title=['Efficiency', 'Rejection'], figname=f'{fig_dir}/ROC3.png', is_log='y')
    
    # confusion matrix
    cm = metrics.confusion_matrix(labels, np.argmax(logits, axis=1))
    ncm = cm / np.sum(cm, axis=1, keepdims=True)
    mkdir(f'{fig_dir}/cm')
    plot_confution_matrix(cm,  xlabels, figname=f'{fig_dir}/cm/raw.png', norm=-1)
    plot_confution_matrix(ncm, xlabels, figname=f'{fig_dir}/cm/purity.png', norm=0)
    plot_confution_matrix(ncm, xlabels, figname=f'{fig_dir}/cm/efficiency.png', norm=1)
    
    
    mkdir(f'{fig_dir}/hist')
    masks = {key: labels == i for i, key in enumerate(xlabels)}
    for i, xlabel in enumerate(xlabels):
        p = scores[:, i]
        hists = {key:p[mask] for key, mask in masks.items()}
        figname = f'{fig_dir}/hist/plot_{xlabel}.png'
        plot_hist(hists, [f'score for {xlabel}', 'count'], figname, 50, [0,1], is_log='y', norm=True)

def make_comparison_results(workdirs, model_name, worklabels, comp_dir, results=None, xlabels=None):
    
    metrics = ['loss', 't1_acc', 't2_acc']
    histories = {}
    for workdir, worklabel in zip(workdirs,worklabels):
        histories[worklabel] = History(f'{workdir}/history-{model_name}.json', {k:var_book(k, ['test'], 0) for k in metrics })
    
    for metric in metrics:
        p = {}
        print(metric)
        for wlabel, history in histories.items():
            x, y = history.metrics[metric].get_value('test')
            title = history.metrics[metric].titles()
            p[wlabel] = [x,y]
        plot_simple(p, title, f'{comp_dir}/comp_{metric}.png')
    
    if not results:
        return 
    
    # roc curve
    roc = {}
    for workdir, worklabel in zip(workdirs,worklabels):
        data = np.load(f'{workdir}/eval-outputs.npz', allow_pickle=True)
        logits, labels = data['logit'], data['label'].astype(int)
        if xlabels is None:
            xlabels = make_labels(data['label_table'], data['label_name'])
            
        scores = transform_score(softmax(logits))
        # roc curve
        roc[worklabel] = make_roc_curve(scores, labels, xlabels)
    
    for l in xlabels:
        roc1, roc2 = {}, {}
        for w in worklabels:
            fpr = roc[w][l]['fpr']
            tpr = roc[w][l]['tpr']
            auc = roc[w][l]['auc']
            roc1[f'{w}:AUC={auc:.3f}'] = [tpr, fpr]
            roc2[f'{w}:AUC={auc:.3f}'] = [tpr, 1./fpr]
        
        plot_simple(roc1, title=['True positive', 'False positive'], figname=f'{comp_dir}/ROC1_{l}.png')
        plot_simple(roc1, title=['True positive', 'False positive'], figname=f'{comp_dir}/ROC2_{l}.png', is_log='y')
        plot_simple(roc2, title=['Efficiency', 'Rejection'], figname=f'{comp_dir}/ROC3_{l}.png', is_log='y')

def main(opts):
    xlabels = ['QCD', 'Top']
    xlabels = ['ss pair', 'cc pair', 'bb pair', 'gluon', 'ud-quark', 's-quark', 'c-quark', 'b-quark']
    
    metrics = ['loss', 't1_acc', 't2_acc', 'g_norm', 'p_norm', 'lr']
    if opts.MoE:
        metrics += ['load_loss', 'imp_loss', 'expert_label_loss']
    
    if 'nominal' in opts.mode:
        plot_all_history(opts.workdir[0], opts.model_name, opts.burn_in, metrics)
    elif 'result' in opts.mode:
        make_results(opts.workdir[0], opts.model_name, xlabels=xlabels)
    elif 'seed' in opts.mode and opts.comp_dir is not None:
        mkdir(opts.comp_dir)
        plot_multi_history(opts.comp_dir, opts.workdir, opts.model_name, opts.burn_in, metrics)
    elif 'weight' in opts.mode:
        plot_all_weight_history(opts.workdir[0])
    elif opts.mode in ['lr-test', 'lr-range-test']:
        plot_lr_range_test(opts.workdir[0], metrics, opts.model_name)
    elif 'comp' in opts.mode and opts.comp_dir is not None:
        print(opts.comp_dir)
        mkdir(opts.comp_dir)
        make_comparison_results(opts.workdir, opts.model_name, opts.worklabels, opts.comp_dir, results=True, xlabels=xlabels)
    elif 'comp-old' in opts.mode and opts.comp_dir is not None:
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
    parser.add_argument('-mn', '--model_name', action = 'store', dest = 'model_name', type = str, default = 'nominal')
    parser.add_argument('-w',  '--workdir',    action = 'store', dest = 'workdir',     type = str, nargs = '*')
    parser.add_argument('-wl', '--worklabels', action = 'store', dest = 'worklabels',  type = str, nargs = '*')
    parser.add_argument('-m',  '--mode',     action = 'store', dest = 'mode',         type = str, default = 'nominal')
    parser.add_argument('-cd', '--comp_dir', action = 'store', dest = 'comp_dir',     type = str, default = None)
    parser.add_argument('-bi', '--burn_in',  action = 'store', dest = 'burn_in',      type = int, default = 0)
    parser.add_argument('-moe', '--moe',     action = 'store_true', dest = 'MoE')
    opts = parser.parse_args()
    main(opts)
    
    
    
    
    
    
    
    
    
    