
import os

import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#colors = np.array(colors)

from color import nord_color
colors = [nord_color.color(c) for c in ('red', 'green', 'blue', 'yellow', 'violet', 'dark3', 'orange', 'frost light blue') ]
colors = [nord_color.color(c) for c in ('frost green', 'green', 'yellow', 'orange', 'red',
                                        'violet', 'frost light blue',  'blue', 'light0') ]

cmap = plt.cm.get_cmap('nord_rainbow', 100)

markers = ['o', ',', 'v', '^', '<', '>']

def get_hatch(label):
    if label in ['b-jet', 'c-jet']:
        return '//'
    elif label in ['u-jet', 'd-jet', 'ud-jet']:
        return '\\\\'
    elif label in ['g-jet',]:
        return None
    return None

def plot_confution_matrix(_cm, labels, figname, norm = -1 ):
    #fig = plt.figure(figsize=(8.0*1.06, 8.0) )
    fig = plt.figure(figsize=(6.0*1.06, 6.0) )
    gs = fig.add_gridspec(nrows = 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    cmap = 'nord_mono_blue'
    
    if norm >= 0:
        cm = _cm.astype('float') / _cm.sum(axis=norm, keepdims = True) * 100.0
    else:
        cm = _cm
    im = ax.imshow(cm, interpolation='nearest', cmap = cmap)
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.08)
    # fig.colorbar(im, cax = cax)
    
    # We want to show all ticks...
    ax.set(xticks = np.arange(cm.shape[1]), yticks = np.arange(cm.shape[0]),
            xticklabels = labels, yticklabels = labels,
            ylabel = 'True label', xlabel = 'Predicted label',
            title = 'Efficiency' if norm == 1 else 'Purity' if norm == 0 else None)
    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label',      fontsize=16)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '2f' 
    thresh = cm.max() / 2. if norm < 0 else 50.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.1f}%' if norm >= 0 else f'{cm[i,j]}', 
                    ha = "center", va = "center", color = "white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.xlim(-0.5, len(labels)-0.5)
    plt.ylim(len(labels)-0.5, -0.5)
    
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_curves(X, Y, Axis, labels, figname, is_log = None):
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    gs = fig.add_gridspec(nrows = 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    
    for i, (x, y, label) in enumerate(zip(X, Y, labels)):
        ax.plot(x, y, '-', alpha = 1.0, label = label, color = colors[i])
    
    ax.set_xlabel(Axis[0], fontsize=16)
    ax.set_ylabel(Axis[1], fontsize=16)
    ax.grid()
    ax.legend()
    ax.minorticks_on()
    
    if is_log is not None:
        if is_log[0] :
            ax.set_xscale("log")
        if is_log[1] :
            ax.set_yscale("log")
    
    
    #plt.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_2d_hist(X, Y, var1, var2, figname, is_norm_y = False, remove_0 = True):
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    
    gs = fig.add_gridspec(ncols = 3, nrows = len(X.keys()), hspace=0.1, wspace=0.05)
    ax = gs.subplots(sharex=True, sharey=True)
    
    from matplotlib.colors import LogNorm
    for i, key in enumerate(X.keys()):
        x = X[key]
        y = Y[key]
        # hist, xedges, yedges, images = ax[i].hist2d(x, y, bins = [var1.bins, var2.bins], range = [var1.range, var2.range], cmap = 'nord_rainbow') #, norm = LogNorm())
        hist, xedges, yedges = np.histogram2d(x, y, bins = [var1.bins_array(), var2.bins_array()])
        extent = [xedges.min(), xedges.max(), yedges.min(), yedges.max()]
        if remove_0:
            hist = np.ma.masked_where(hist == 0 , hist)
        
        
        hist_x = hist/np.max(hist, axis = 0, keepdims = True)
        hist_y = hist/np.max(hist, axis = 1, keepdims = True)
        
        im = ax[i,0].imshow(hist.T,   origin='lower', extent = extent, cmap = 'nord_rainbow', aspect = 'auto')
        im = ax[i,1].imshow(hist_x.T, origin='lower', extent = extent, cmap = 'nord_rainbow', aspect = 'auto')
        im = ax[i,2].imshow(hist_y.T, origin='lower', extent = extent, cmap = 'nord_rainbow', aspect = 'auto')
        
        #ax[i].scatter(x, y, alpha = 0.25, label = key)
        
        for j in range(3):
            ax[i, j].tick_params(axis='x', labelsize= 8)
            ax[i, j].tick_params(axis='y', labelsize= 8)
            ax[i, j].minorticks_on()
        #fig.colorbar(images,ax=ax[i])
        ax[i,0].set_ylabel(f'{var2.label}({key})',     fontsize = 8)
        
    ax[-1,0].set_xlabel(f'{var1.label}',            fontsize = 8)
    ax[-1,1].set_xlabel(f'{var1.label}(profile x)', fontsize = 8)
    ax[-1,2].set_xlabel(f'{var1.label}(profile y)', fontsize = 8)
    
        
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_2d_scatter(X, Y, var1, var2, figname):
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    
    gs = fig.add_gridspec(ncols = 1, nrows = 1, hspace=0)
    ax = gs.subplots(sharex = False, sharey = False)
    
    
    from matplotlib.colors import LogNorm
    for i, key in enumerate(X.keys()):
        x = X[key]
        y = Y[key]
        #hist, xedges, yedges, images = ax[i].hist2d(x, y, bins = [var1.bins, var2.bins], range = [var1.range, var2.range], cmap = 'nord_rainbow', norm = LogNorm())
        ax.scatter(x, y, alpha = 0.25, label = key, color = colors[i], marker = markers[i])
        ax.set_xlabel(var1.label, fontsize = 20)
        ax.set_ylabel(var2.label, fontsize = 20)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_1d(X, var, figname, is_norm = False):
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    gs = fig.add_gridspec(nrows = 1, ncols = 1, hspace=0)
    ax = gs.subplots(sharex=False, sharey=False)
    
    from matplotlib.colors import LogNorm
    
    for i, (key, val) in enumerate(X.items()):
        weights = np.ones(len(val))/float(len(val)) if is_norm else None
        hist, xedges, _ = ax.hist(val, bins = var.bins, label = key, range = var.range,
                                    weights = weights, 
                                    alpha = 1.0, 
                                    #histtype = 'stepfilled', 
                                    histtype = 'step', 
                                    
                                    facecolor = colors[i],
                                    # hatch = get_hatch(key),
                                    zorder=0,
                                    edgecolor = colors[i], 
                                    #edgecolor = nord_color.color('dark0'), 
                                    linewidth = 1.5)
        if var.is_log:
            ax.set_yscale('log')
    from matplotlib.ticker import AutoMinorLocator
    #ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlim( var.range )
    #ax.minorticks_on()
    plt.xlim(*var.range)
    
    ax.set_xlabel(var.label, fontsize = 20)
    ax.set_ylabel('count', fontsize = 20)
    ax.grid()
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_1d_eff(X, Y, var, figname, is_norm = False):
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    gs = fig.add_gridspec(nrows = 1, ncols = 1, hspace=0)
    ax = gs.subplots(sharex=False, sharey=False)
    
    from matplotlib.colors import LogNorm
    
    for i, (key, val) in enumerate(X.items()):
        ax.plot(Y[key], val, label = key, 
                                    alpha = 1.0, 
                                    color = colors[i],
                                    linewidth = 1.5)
    ax.set_xlim( var.range )
    ax.set_ylim((0.0, 1.0))
    plt.xlim(*var.range)
    plt.ylim(0.0, 1.0)
    
    
    ax.set_xlabel(var.label,    fontsize = 20)
    ax.set_ylabel('efficiency', fontsize = 20)
    ax.grid()
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def make_MoE_plot(labels, ew, figname):
    
    fig = plt.figure(figsize=(6.0, 6.0) )
    gs = fig.add_gridspec(nrows = 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    cmap = 'nord_mono_blue'
    
    im = ax.imshow(ew, interpolation='nearest', cmap = cmap)
    
    # We want to show all ticks...
    ax.set(xticks = np.arange(ew.shape[1]), yticks = np.arange(ew.shape[0]),
            #xticklabels = [f'{i}' for i in range(ew.shape[0])], yticklabels = labels,
            title = 'Average Expert Weight' )
    ax.set_xlabel('Expert Index', fontsize=16)
    ax.set_ylabel('Class Label',  fontsize=16)
    
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '2f' 
    for i in range(ew.shape[0]):
        for j in range(ew.shape[1]):
            ax.text(j, i, f'{ew[i, j]:.3f}', ha = "center", va = "center", color = "black")
    
    fig.tight_layout()
    #plt.xlim(-0.5, len(labels)-0.5)
    #plt.ylim(len(labels)-0.5, -0.5)
    
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def softmax(x):
    max = np.max(x,   axis=1, keepdims=True)
    e_x = np.exp(x - max) 
    sum = np.sum(e_x, axis=1, keepdims=True) 
    f_x = e_x / sum 
    return f_x

class Variables:
    def __init__(self, masks = None):
        self.vars = {}
        self.masks = masks

    def keys(self):
        return self.vars.keys()

    def register(self, **kwargs):
        tmp = Variable(**kwargs)
        self.vars[tmp.name] = tmp
        
    def __getitem__(self, key):
        return self.vars[key]
    
    def plot_1d(self, figname, is_norm = True):
        for var in self.vars.keys():
            v = self.vars[var]
            print(var, v.values.shape)
            print(self.masks['g-jet'].shape)
            X = { k:v.values[mask] for k, mask in self.masks.items()}
            plot_1d(X, v, figname = f'{figname}/{v.name}.png', is_norm = is_norm)
    
    def plot_2d(self, figname):
        import itertools
        combinations = list(itertools.combinations(self.vars.keys(), 2))
        for var1, var2 in combinations:
            v1 = self.vars[var1]
            v2 = self.vars[var2]
            X = { k:v1.values[mask] for k, mask in self.masks.items()}
            Y = { k:v2.values[mask] for k, mask in self.masks.items()}
            plot_2d_hist(X, Y, v1, v2, figname = f'{figname}/{v1.name}.{v2.name}.png')
    
    def plot_1d_eff(self, probs, figname, is_norm = True):
        probs = np.array(probs).T
        for var in self.vars.keys():
            v = self.vars[var]
            X = {}
            Y = {}
            
            for idx, (flavor, mask) in enumerate(self.masks.items()):
                mask2 = np.argmax(probs, axis = 1) == idx
                hist_all, e = np.histogram(v.values[mask], bins = v.bins, range = v.range)
                hist_eff, e = np.histogram(v.values[mask2 & mask], bins = v.bins, range = v.range)
                
                X[flavor] = hist_eff / hist_all
                y = []
                for y1, y2 in zip(e[:-1], e[1:]):
                    y.append((y1 + y2)*0.5)
                Y[flavor] = np.array(y)
                plot_1d_eff(X, Y, v, figname = f'{figname}/{v.name}-eff.png', is_norm = is_norm)

class Variable:
    def __init__(self, name, label, bins, range, values, is_log = False, norm = False):
        self.name = name
        self.label = label
        self.bins = bins
        self.range = range
        self.values = values
        self.is_log = is_log
        self.norm = norm
        if self.norm:
            self.range = (-1.0, +1.0)
        
    def bins_array(self):
        return np.linspace( self.range[0], self.range[1], self.bins+1)

def transform_score(x):
    xmin, xmax = np.min(x), np.max(x)
    print(xmin, xmax)
    return (x - xmin)/(xmax - xmin)
    #return x
    
def tagging_score(prob, idx):
    return prob[:, idx]/(np.sum(prob, axis=1) - prob[:, idx])

def open_memmap(filename, shape):
    fp = np.memmap(filename, dtype = 'float32', mode = 'r', shape = shape)
    x = fp[:]
    del fp
    return x

def open_npz(filename, xlist = ['logits', 'labels']):
    data = np.load(filename, allow_pickle = True)
    #props  = data['props']
    return [data[x] for x in xlist]

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def make_roc_curve(jet_labels, probs, labels, fig_dir):
    mkdir(fig_dir)
    from sklearn import metrics
    
    one_hot = np.eye(len(jet_labels))[labels]
    X, Y, R, L = [], [], [], []
    for i in range(len(jet_labels)):
        #print(one_hot[:,i].shape)
        fpr, tpr, thresholds = metrics.roc_curve(one_hot[:,i], probs[i])
        AUC = metrics.auc(fpr, tpr)
        X.append(tpr)
        Y.append(fpr)
        R.append(1.0/fpr)
        L.append(f'{jet_labels[i]}:AUC={AUC:.2f}')
    
    plot_curves(X = X, Y = Y, Axis = ['True positive', 'False positive'], labels = L, figname = f'{fig_dir}/ROC1.png')
    plot_curves(X = X, Y = Y, Axis = ['True positive', 'False positive'], labels = L, figname = f'{fig_dir}/ROC2.png', is_log = [False,True])
    plot_curves(X = X, Y = R, Axis = ['Efficiency', 'Rejection'], labels = L, figname = f'{fig_dir}/ROC3.png', is_log = [False,True])
    
    X, Y, L = [], [], []
    for i in range(len(jet_labels)):
        precision, recall, thresholds = metrics.precision_recall_curve(one_hot[:,i], probs[i])
        AP = metrics.average_precision_score(one_hot[:,i], probs[i])
        X.append(precision)
        Y.append(recall)
        L.append(f'{jet_labels[i]}:AP={AP:.2f}')
    
    plot_curves(X = X, Y = Y, Axis = ['Precision', 'Recall'], labels = L, figname = f'{fig_dir}/ROC4.png')

def jet_to_label(jet):
    if '-jet' in jet:
        jet = jet.replace('-jet', '')
    
    labels = []
    jet_table = {'g':[0], 'd':[1], 'u':[2], 's':[3], 'c':[4], 'b':[5], 'l':[1,2,3]}
    for f in jet:
        if f not in jet_table.keys():
            raise(f'{f}-jet is not in the tabel {jet_table}')
        labels += jet_table[f]
    return labels

def make_roc_curve_2(probs, labels, sig_label, bkg_labels, fig_dir):
    mkdir(fig_dir)
    from sklearn import metrics
    
    X, Y, R, L = [], [], [], []
    sig = jet_to_label(sig_label)
    
    
    
    
    for i, bkg_label in enumerate(bkg_labels):
        bkg = jet_to_label(bkg_label)
        
        this_labels = np.zeros_like(labels)
        
        sig_idx = labels < -1
        
        for s in sig:
            tmp = labels == s
            sig_idx = sig_idx | tmp
        
        this_labels[sig_idx] = 1
        
        bkg_idx = labels < -1
        for b in bkg:
            tmp = labels == b
            bkg_idx = bkg_idx | tmp
        
        n_sig = len(this_labels[sig_idx])
        n_bkg = len(this_labels[bkg_idx])
        print(bkg_label)
        print(bkg)
        weights = np.ones(this_labels.shape )
        print(weights.shape)
        
        weights[sig_idx] = float(n_sig+n_bkg) / float(n_sig)
        weights[bkg_idx] = float(n_sig+n_bkg) / float(n_bkg)
        
        all_idx = bkg_idx | sig_idx
        
        this_probs = np.sum(probs[s][all_idx] for s in sig)
        this_labels = this_labels[all_idx]
        weights = weights[all_idx]
        
        fpr, tpr, thresholds = metrics.roc_curve(y_true = this_labels, pos_label = 1, y_score = this_probs, sample_weight = weights)
        AUC = metrics.auc(fpr, tpr)
        X.append(tpr)
        Y.append(fpr)
        R.append(1.0/fpr)
        L.append(f'{bkg_label}:AUC={AUC:.2f}')
        
    plot_curves(X = X, Y = Y, Axis = ['True positive', 'False positive'], labels = L, figname = f'{fig_dir}/ROC1.png')
    plot_curves(X = X, Y = Y, Axis = ['True positive', 'False positive'], labels = L, figname = f'{fig_dir}/ROC2.png', is_log = [False,True])
    plot_curves(X = X, Y = R, Axis = ['Efficiency', 'Rejection'], labels = L, figname = f'{fig_dir}/ROC3.png', is_log = [False,True])
    pass

import jax
import jax.numpy as jnp
def naive_top_k(data, k, axis = -1):
    N = data.shape[axis]
    def top1(data_, unused):
        index = jnp.argmax(data_, axis=axis)
        mask = jax.nn.one_hot(index, N, axis = axis, dtype=jnp.bool_)
        data_wo_top1 = data_ * (1 - mask)
        return data_wo_top1, index
    index = []
    for i in range(k):
        data, idx = top1(data, None)
        index.append(idx)
        
    index = jnp.stack(index)
    index = jnp.moveaxis(index, 0, -1)
    return index

def main(opts):
    
    norm = False
    #norm = True
    
    
    jet_labels = ['g-jet', 'd-jet', 'u-jet', 's-jet', 'c-jet', 'b-jet'] 
    workdir = opts.workdir
    print(workdir)
    fig_dir = f'{workdir}/figs'
    print(fig_dir)
    mkdir(fig_dir)
    mkdir(f'{fig_dir}/hist1d')
    mkdir(f'{fig_dir}/hist2d')

    
    
    
    logits, labels = open_npz(f'{workdir}/eval-outputs.npz')
    labels = labels.astype(int)
    print('logits.shape', logits.shape)
    print('labels.shape', labels.shape)
    
    expert_weights = None
    if opts.isMoE:
        expert_weights = open_npz(f'{workdir}/eval-outputs.npz', xlist=['expert_weights'])[0]
        print('expert_weights.shape', expert_weights.shape)
        mkdir(f'{fig_dir}/MoE')
    
    scores = softmax(sigmoid(logits))
    print(scores.shape)
    probs = [transform_score(scores[:, i]) for i in range(len(jet_labels))]
    masks = { key: labels == i for i, key in enumerate(jet_labels) } 
    
    vars = Variables(masks)
    for i, jet_label in enumerate(jet_labels):
        vars.register(name = f'score-{jet_label}',   label = f'{jet_label} score',    bins = 50, range = [ 0,1.0], values = probs[i],   is_log = True)
    
    
    #vars.plot_1d(figname = f'{fig_dir}/hist1d')
    #vars.plot_2d(figname = f'{fig_dir}/hist2d')
    
    # ROC
    # make_roc_curve(jet_labels, probs, labels, fig_dir = f'{fig_dir}/ROC/all')
    
    # make_roc_curve_2(probs, labels, sig_label = 'b-jet',  bkg_labels = ['c-jet', 'udsg-jet', 'udsgc-jet'],      fig_dir = f'{fig_dir}/ROC/b-jet')
    # make_roc_curve_2(probs, labels, sig_label = 'c-jet',  bkg_labels = ['b-jet', 'udsg-jet', 'udsgb-jet'],      fig_dir = f'{fig_dir}/ROC/c-jet')
    # make_roc_curve_2(probs, labels, sig_label = 'g-jet',  bkg_labels = ['ud-jet', 'uds-jet', 'b-jet', 'c-jet'], fig_dir = f'{fig_dir}/ROC/g-jet')
    # make_roc_curve_2(probs, labels, sig_label = 's-jet',  bkg_labels = ['ud-jet', 'g-jet', 'udg-jet'],          fig_dir = f'{fig_dir}/ROC/s-jet')
    # make_roc_curve_2(probs, labels, sig_label = 'l-jet',  bkg_labels = ['g-jet', 'b-jet', 'c-jet'],             fig_dir = f'{fig_dir}/ROC/l-jet')
    topK = 6
    if expert_weights is not None:
        n_layer = expert_weights.shape[1]
        print(expert_weights.shape)
        for idx in range(n_layer):
            ew = expert_weights[:, idx]
            print(idx)
            print('ew  : ', ew.shape)
            #ew, _ = jax.lax.top_k(ew, k=1)
            ew = np.sum(ew, axis=1)/ew.shape[1]
            
            #ew = np.mean(ew, axis = 1)
            
            print(ew[10])
            input('Enter')
            
            
            
            
            ews = []
            for i in range(len(jet_labels)):
                mask = labels == i
                ews.append(np.mean(ew[mask], axis=0))
            ew = np.stack(ews)
            print(ew.shape)
            make_MoE_plot(jet_labels, ew, figname = f'{fig_dir}/MoE/expert_weights_layer{idx}.png')
    
    # confusion matrix
    from sklearn.metrics import confusion_matrix
    pred = np.argmax(logits, axis = 1)
    cm = confusion_matrix(labels, pred)
    print(cm)
    
    
    normed_cm = cm / np.sum(cm, axis = 1, keepdims = True)
    
    print(normed_cm)
    mkdir(f'{fig_dir}/cm')
    plot_confution_matrix(cm, jet_labels, figname = f'{fig_dir}/cm/raw.png', norm = -1 )
    plot_confution_matrix(normed_cm, jet_labels, figname = f'{fig_dir}/cm/purity.png', norm = 0 )
    plot_confution_matrix(normed_cm, jet_labels, figname = f'{fig_dir}/cm/efficiency.png', norm = 1 )
    
def main_with_prop(opts):
    
    norm = False
    #norm = True
    
    
    jet_labels = ['ud-jet', 's-jet', 'c-jet', 'b-jet', 'g-jet', 'gud-jet', 'gs-jet', 'gc-jet', 'gb-jet'] 
    #jet_labels = ['uds-jet', 'c-jet', 'b-jet', 'g-jet', 'guds-jet', 'gc-jet', 'gb-jet'] 
    jet_labels = ['g-jet', 'u-jet', 'd-jet', 's-jet', 'c-jet', 'b-jet'] 

    #jet_labels = ['uds-jet', 'c-jet', 'b-jet', 'g-jet'] 
    workdir = opts.workdir
    print(workdir)
    fig_dir = f'{workdir}/figs'
    print(fig_dir)
    mkdir(fig_dir)
    mkdir(f'{fig_dir}/hist1d')
    mkdir(f'{fig_dir}/hist1d-eff')
    mkdir(f'{fig_dir}/hist2d')
    
    logits, labels, props = open_npz(f'{workdir}/eval-outputs.npz')
    labels = labels.astype(int)
    print(logits.shape)
    print(labels.shape)
    print(props.shape)
    
    
    
    
    pt, n_charged, n_neutral = props[:, 0], props[:, 1], props[:, 2]#, props[:, 3]
    
    
    scores = softmax(sigmoid(logits))
    
    probs = [transform_score(scores[:, i]) for i in range(len(jet_labels))]
    
    masks = { key: labels == i for i, key in enumerate(jet_labels) } 
    
    vars = Variables(masks)
    vars.register(name = 'n_charged', label = r'#of charged',           bins = 20, range = [ 0, 20], values = n_charged, norm = norm)
    vars.register(name = 'n_neutral', label = r'#of neutral',           bins = 20, range = [ 0, 20], values = n_neutral, norm = norm)
    vars.register(name = 'pt',        label = r'$p_{\mathrm{T}}$[GeV]', bins = 40, range = [ 0,400], values = pt, is_log = True, norm = norm)
    #vars.register(name = 'eta',       label = r'$\eta$',                bins = 60, range = [-3,  3], values = eta, norm = norm)
    for i, jet_label in enumerate(jet_labels):
        #vars.register(name = f'out-{jet_label}',     label = f'{jet_label} out',      bins = 50, range = [-7.5,1.0], values = outputs[:, i], is_log = True)
        #vars.register(name = f'sigmoid-{jet_label}', label = f'{jet_label} sigmoid',  bins = 50, range = [-1.0,1.0], values = _outputs2[:, i], is_log = True)
        vars.register(name = f'score-{jet_label}',   label = f'{jet_label} score',    bins = 50, range = [ 0,1.0], values = probs[i],   is_log = True)
    
    
    vars.plot_1d(figname = f'{fig_dir}/hist1d')
    vars.plot_2d(figname = f'{fig_dir}/hist2d')

    vars2 = Variables(masks)
    
    vars2.register(name = 'n_charged', label = r'#of charged',           bins = 20, range = [ 0, 20], values = n_charged, norm = norm)
    vars2.register(name = 'n_neutral', label = r'#of neutral',           bins = 20, range = [ 0, 20], values = n_neutral, norm = norm)
    #vars2.register(name = 'eta',       label = r'$\eta$',                bins = 60, range = [-3,  3], values = eta, norm = norm)
    vars2.register(name = 'pt',        label = r'$p_{\mathrm{T}}$[GeV]', 
                    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 300, 400],
                    range = [ 0,400], values = pt, is_log = True, norm = norm)
    vars2.plot_1d_eff(probs, figname = f'{fig_dir}/hist1d-eff')
    
    
    # ROC
    make_roc_curve(jet_labels, probs, labels, sigs = None, bkgs = None, fig_dir = f'{fig_dir}/ROC/all')
    
    make_roc_curve_2(['c-jet', 'udsg-jet', 'udsgc-jet'], probs, labels, sigs = [(4,), (4,), (4,)], bkgs = [(3,), (0,1,2), (0,1,2,3)], fig_dir = f'{fig_dir}/ROC/b-jet')
    make_roc_curve_2(['b-jet', 'udsg-jet', 'udsgb-jet'], probs, labels, sigs = [(3,), (3,), (3,)], bkgs = [(4,), (0,1,2), (0,1,2,4)], fig_dir = f'{fig_dir}/ROC/c-jet')
    make_roc_curve_2(['ud-jet', 'uds-jet', 'b-jet', 'c-jet'], probs, labels, sigs = [(0,), (0,)],   bkgs = [(1,), (1,2), (4,), (3,)], fig_dir = f'{fig_dir}/ROC/g-jet')
    make_roc_curve_2(['ud-jet', 'g-jet', 'udg-jet'],     probs, labels, sigs = [(1,), (1,)],   bkgs = [(1,), (1,2)], fig_dir = f'{fig_dir}/ROC/s-jet')
    make_roc_curve_2(['g-jet', 'b-jet', 'c-jet'],        probs, labels, sigs = [(1,2), (1,2), (1,2)], bkgs = [(0,), (4,), (3,) ], fig_dir = f'{fig_dir}/ROC/l-jet')
    
    
    
    
    # confusion matrix
    from sklearn.metrics import confusion_matrix
    pred = np.argmax(logits, axis = 1)
    cm = confusion_matrix(labels, pred)
    print(cm)
    
    
    normed_cm = cm / np.sum(cm, axis = 1, keepdims = True)
    
    print(normed_cm)
    mkdir(f'{fig_dir}/cm')
    plot_confution_matrix(cm, jet_labels, figname = f'{fig_dir}/cm/raw.png', norm = -1 )
    plot_confution_matrix(normed_cm, jet_labels, figname = f'{fig_dir}/cm/purity.png', norm = 0 )
    plot_confution_matrix(normed_cm, jet_labels, figname = f'{fig_dir}/cm/efficiency.png', norm = 1 )
    
    

class JetFeature:
    def __init__(self, sample, path, n_files, shape = (65536, 6)):
        self.sample = sample
        self.path = path
        self.n_files = n_files
        self.shape = shape
        self.vars = ['pt', 'eta', 'phi', 'e', 'n_charged', 'n_neutral']
    
    def val(self, var):
        if var not in self.vars:
            raise ValueError(f'{var} is not in {self.vars}')
        
        qidx = self.vars.index(var)
        return self.get(qidx)
    
    def get(self, qidx):
        x = []
        query_files = [f'{self.path}/{self.sample}/{name}' for name in os.listdir(f'{self.path}/{self.sample}') if name.startswith(f'{self.sample}-query')][:self.n_files]
        for file in query_files:
            fp = np.memmap(file, dtype = 'float32', mode = 'r', shape = self.shape)
            x.append(fp[:, qidx])
            del fp
        x = np.concatenate(x, axis = 0)
        return x
    
def plot_jet_attr(opts):
    
    ver = '2021-12-01'
    save_dir = f'figs/jet-query/{ver}'
    mkdir(save_dir)
    mkdir(f'{save_dir}/hist1d')
    
    
    
    
    samples = ['g-jet', 'd-jet', 'u-jet', 's-jet', 'c-jet', 'b-jet']
    
    jets = { sample:JetFeature(sample, path = f'data/{ver}', n_files = opts.nfiles) for sample in samples}
    
    
    vars = Variables()
    vars.register(name = 'pt',     label = r'$p_{\mathrm{T}}$[GeV]', bins = 50, range = [ 0,500], values = None, is_log = True)
    vars.register(name = 'eta',    label = r'$\eta$',                bins = 60, range = [-3,  3], values = None, is_log = False)
    vars.register(name = 'phi',    label = r'$\phi$',                bins = 64, range = [-3.2, 3.2], values = None, is_log = False)
    vars.register(name = 'e',      label = r'$E$[GeV]',              bins = 50, range = [0, 500], values = None, is_log = True)
    vars.register(name = 'n_charged', label = r'#of track',          bins = 20, range = [0, 20], values = None, is_log = False)
    vars.register(name = 'n_neutral', label = r'#of cluster',        bins = 20, range = [0, 20], values = None, is_log = False)
    
    
    for var in vars.keys():
        v = vars[var]
        X = { sample: jets[sample].val(v.name) for sample in samples}
        plot_1d(X, v, figname = f'{save_dir}/hist1d/{v.name}.png')

if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s',  '--seed',       action = 'store', dest = 'seed',       type = int, default = 3407)
    # parser.add_argument('-m',  '--model',      action = 'store', dest = 'model',      type = str, default = 'MLPMixer')
    # parser.add_argument('-cl', '--classifier', action = 'store', dest = 'classifier', type = str, default = 'MLP')
    # parser.add_argument('-i',  '--id',         action = 'store', dest = 'id',         type = str, default = None)
    parser.add_argument('-mn', '--model_name', action = 'store', dest = 'model_name', type = str, default = 'nominal')
    parser.add_argument('-w', '--workdir',     action = 'store', dest = 'workdir',    type = str, default = None)
    parser.add_argument('-m', '--isMoE',       action = 'store', dest = 'isMoE',      type = bool, default = True)

    opts = parser.parse_args()
    
    main(opts)
    #plot_jet_attr(opts)



