import torch

import os, math
import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


from color import nord_color

colors = [nord_color.color(c) for c in ('red', 'green', 'blue', 'orange', 'violet', 'frost light blue',) ]
colors = [nord_color.color(c) for c in ('frost green', 'green', 'yellow', 'orange', 'red',
                                        'violet', 'frost light blue',  'blue', 'light0') ]




def mkdir(dir):
    if not os.path.exists(f'{dir}'):
        os.makedirs(f'{dir}')

def set_violin_color(violin, color, lw=1, alpha=0.5):
    for vp in violin['bodies']:
        vp.set_facecolor(color)
        vp.set_edgecolor(color)
        vp.set_linewidth(lw)
        vp.set_alpha(alpha)
    for key in violin.keys():
        if 'bodies' in key: 
            continue
        vp = violin[key]
        vp.set_facecolor(color)
        vp.set_edgecolor(color)
        vp.set_linewidth(lw)
        vp.set_alpha(alpha)
    
def plot_violin(embeddings, figname, offset = 0.2 ):
    fig = plt.figure(figsize=(12.0, 8.0 ) )
    gs = fig.add_gridspec(nrows = 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=False)
    
    violins = {}
    n_layer = len(embeddings)
    n_step = offset * n_layer + 0.4
    cmap = plt.cm.get_cmap('nord_rainbow', 100)
    
    plt.setp(ax.get_xminorticklabels(), visible=False)
    plt.minorticks_off()
    
    for idx, (key, weight) in enumerate(embeddings.items()):
        w = list(weight.values())
        if idx == 0 :
            labels = list(weight.keys())
            
            
            xticks = np.arange(0.0, len(labels)*n_step, n_step)
            print(xticks)
            xticks += offset * (n_layer-1)/ 2.0
            print(xticks)
            ax.set_xticks(xticks, labels=labels, rotation = 45.0, minor = False, ha = 'center')
            ax.set_xlim(-0.5, len(labels)*n_step + 0.0)
            
        
        
        pos = [p+offset*(idx+0) for p in np.arange(0, len(w)*n_step, n_step)]
        print(pos)
        violin = ax.violinplot(w, quantiles=[[0.25, 0.50, 0.75]] * len(w), positions = pos, showmedians=True, showmeans=True)
        #color = colors[idx]
        color = cmap([float(idx/(n_layer-1))])
        
        set_violin_color(violin, color)
        violins[key] = violin['bodies'][0]
    #plt.ylim(-2, 2)
    plt.setp(ax.get_xminorticklabels(), visible=False)
    plt.minorticks_off()
    ax.grid()
    #ax.legend(violins.values(), violins.keys(), ncol = 4)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')


def change_layer_name(layer):
    d = {'module.':'', 'embeddings':'E', 'classifiers':'C', 
        'film_gen.beta.layers':'F-beta', 'film_gen.gamma.layers':'F-gamma', 
        'linear':'linear', 'weight':'w', 'bias':'b'}
    for k, v in d.items():
        layer = layer.replace(k, v)
    return layer

def plot_results( model, classifier, id, seed, is_plot = True) : 
    
    ver = 'ver1'
    filename = f'results/{model}/{classifier}/{id}/{seed}/weights_{ver}.npz'
    
    
    weights = {}
    
    file = np.load(filename, allow_pickle = True)
    n_layer = 0
    n_film = 0
    n_classifier = 0
    for key in list(file.keys()):
        layer = change_layer_name(key)
        weight = file[key]
        weight = weight.reshape(-1)
        weights[layer] = weight
        
        if layer.startswith('E.'):
            l = int(layer.split('.')[1])
            if l >= n_layer:
                n_layer = l
        
        if layer.startswith('F-beta.'):
            l = int(layer.split('.')[1])
            if l >= n_film:
                n_film = l
        
        if layer.startswith('C.'):
            l = int(layer.split('.')[1])
            if l >= n_classifier:
                n_classifier = l
    
    def make_weights_table(layers, name, layer_name):
        ret = {}
        for idx in layers:
            tmp = {}
            for key, val in weights.items():
                if key.startswith(f'{layer_name}.{idx}.'):
                    k = key.replace(f'{layer_name}.{idx}.', '')
                    tmp[k] = val
                
            ret[f'{name}:{idx-1}'] = tmp
        return ret
    
    embeddings  = make_weights_table(range(2, n_layer  +1), name = 'gMLP Block', layer_name = 'E')
    classifiers = make_weights_table(range(0, n_classifier  +1), name = 'Classifier Block', layer_name = 'C')
    FiLM_g = make_weights_table(range(0, n_film  +1), name = 'FiLM(gamma) Block', layer_name = 'F-gamma')
    FiLM_b = make_weights_table(range(0, n_film  +1), name = 'FiLM(beta) Block', layer_name = 'F-beta')
    
    save_dir = f'figs/results/{model}/{classifier}/{id}/{seed}/weights/'
    mkdir(save_dir)
    
    plot_violin(embeddings,  figname = f'{save_dir}/plot_weights_embedding.png')
    plot_violin(classifiers, figname = f'{save_dir}/plot_weights_classifier.png')
    plot_violin(FiLM_g,      figname = f'{save_dir}/plot_weights_FiLM_gamma.png')
    plot_violin(FiLM_b,      figname = f'{save_dir}/plot_weights_FiLM_beta.png')
    
    
    
if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-m', '--model',      action = 'store', dest = 'model',      default = None,  type = str)
    parser.add_argument('-c', '--classifier', action = 'store', dest = 'classifier', default = None,  type = str)
    parser.add_argument('-i', '--id',         action = 'store', dest = 'id',         default = None,  type = str)
    parser.add_argument('-s', '--seed',       action = 'store', dest = 'seed',       default = 3407,  type = int)
    opts = parser.parse_args()
    
    if opts.id is not None:
        plot_results(opts.model, opts.classifier, opts.id, opts.seed)
    else:
        #plot_lr('figs/cos.1.png')
        comparison(opts)




