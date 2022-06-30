

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*10
colors = np.array(colors)

import torch
# from torch_geometric.data import Data

import utils
from color import nord_color
colors = [nord_color.color(c) for c in ('red', 'green', 'blue', 'yellow', 'violet', 'dark3', 'orange', 'frost light blue') ]


def make_graph( data, figname ) : 
    
    import networkx as nx
    import torch_geometric
    
    
    plt.figure(figsize=(8.0, 6.0))
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig, ax = plt.subplots()
    
    phi_list = data.x[:,3].tolist()
    
    root = (data.x[:,0] == 0).nonzero(as_tuple=True)[0].item()
    
    nxg = torch_geometric.utils.to_networkx(data)
    draw_pos = utils.hierarchy_pos(nxg, width = 0.2, vert_gap = 0.2, xcenter = 0.0)
    #draw_pos = utils.particle_diagram_pos(nxg, phi_list, root = root, center = (0.0, 0.0), rphi = False)
    
    node_color = []
    labels = {}
    for i, feature in enumerate(data.x):
        pdgId = feature[0]
        plabel = utils.particle_label(pdgId)
        pt = feature[1]
        labels[i] = f'{plabel}\n{pt:.1f}GeV'
        node_color.append(utils.get_particle_color(pdgId))
    
    
    
    nx.draw_networkx_nodes(nxg, draw_pos, node_size = 1000, node_color = node_color, alpha=0.5)
    nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='->', alpha=0.9, arrows = True)
    nx.draw_networkx_labels(nxg, draw_pos, font_size=10, labels = labels )
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.grid()
    #plt.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    
def main(opts):
    
    edge_index = np.load(f'{opts.data_path}/edge_index_{opts.idx:04d}.npz', allow_pickle = True)['data']
    features   = np.load(f'{opts.data_path}/features_{opts.idx:04d}.npz'  , allow_pickle = True)['data']
    
    for i in range(10):
        e = np.array( edge_index[i].tolist() ).astype(np.int64)
        edge_idx = torch.tensor(e, dtype = torch.long)
        feature = torch.tensor(features[i], dtype=torch.float)
        data = Data(x = feature, edge_index = edge_idx)
        
        make_graph(data, f'figs/graph.{i}.png')

def weight5(A,B,G):
    weights = np.zeros((A.shape[0], 4))
    weights[:, 0] = (1.0 - B)*(1.0 - G)*A/2 + A/2
    weights[:, 1] = (1.0 - A)*(1.0 - G)*B/2 + B/2
    weights[:, 2] = (1.0 - A)*(1.0 - B)*G/2 + G/2
    weights[:, 3] = (1.0 - A)*(1.0 - B)*(1.0 - G) - A*B*G/2
    return weights

def weight4(A,B,G):
    weights = np.zeros((A.shape[0], 4))
    weights[:, 0] = (1.0 - B)*(1.0 - G)*A + B*G
    weights[:, 1] = (1.0 - A)*(1.0 - G)*B + A*G
    weights[:, 2] = (1.0 - A)*(1.0 - B)*G + A*B
    weights[:, 3] = (1.0 - A)*(1.0 - B)*(1.0 - G) - 2.0*A*B*G
    return weights

def weight3(A,B,G, inverse=False):
    
    weights = np.zeros((A.shape[0], 4))
    weights[:, 0] = (1.0 - B/2.0)*(1.0 - G/2.0)*A - A*B*G/4.0
    weights[:, 1] = (1.0 - A/2.0)*(1.0 - G/2.0)*B - A*B*G/4.0
    weights[:, 2] = (1.0 - A/2.0)*(1.0 - B/2.0)*G - A*B*G/4.0
    weights[:, 3] = (1.0 - A)*(1.0 - B)*(1.0 - G) + A*B*G
    if inverse :
        weights = 1.0 - weights
    return weights

def weight2(A,B,G):
    weights = np.zeros((A.shape[0], 4))
    weights[:, 0] = A * B
    weights[:, 1] = A * (1.0 - B)
    weights[:, 2] = (1.0 - A)*G
    weights[:, 3] = (1.0 - A)*(1.0 - G)
    return weights

def weight1(A,B,G):
    weights = np.zeros((A.shape[0], 4))
    weights[:, 0] = A
    weights[:, 1] = (1.0 - A)*B
    weights[:, 2] = (1.0 - A)*(1.0 - B)*G
    weights[:, 3] = (1.0 - A)*(1.0 - B)*(1.0 - G)
    return weights


def transform(x):
    #x = 1 - np.exp(x)
    #x = np.log(1-x)
    x = 2.0 * x - 1.0
    x = 1. / (1. + np.exp(-x))
    # x = np.tanh(x) /2.0
    return x

def Sampling(N, P1 = 0.1, P2 = 0.1):
    import torch
    #m = torch.distributions.kumaraswamy.Kumaraswamy(torch.tensor([0.5]), torch.tensor([0.5]))
    m = torch.distributions.beta.Beta(torch.tensor([P1]), torch.tensor([P2]))
    x = m.sample(torch.Size([N,3]))
    x = torch.reshape(x, (N,3))
    return x.numpy()

def Beta(N, p1, p2):
    Lambda = Sampling(N, P1 = p1, P2 = p2)
    weights = weight3(Lambda[:,0], Lambda[:,1], Lambda[:,2], inverse=False)
    return weights


def Dirichlet(N, Ps):
    m = torch.distributions.dirichlet.Dirichlet(torch.tensor(Ps))
    Lambda = m.sample(torch.Size([N]))
    Lambda = Lambda.numpy()
    #print(Lambda.shape)
    return Lambda

def plot_weights():
    import matplotlib.cm as cm
    figname = 'figs/test/weight1.png'
    Ps = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.]
    Ps = [0.01, 0.05, 0.1]
    
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    #gs = fig.add_gridspec(nrows = 1, ncols = len(Ps), hspace=0)
    gs = fig.add_gridspec(nrows = len(Ps), ncols = 1, hspace=0)
    ax = gs.subplots(sharex=False, sharey=False)
    N = 10000
    
    
    # Lambda = np.random.exponential(scale = 1.0, size = (N, 3))
    #Lambda = transform(Lambda)
    
    for i, p in enumerate(Ps):
        weights = Dirichlet(N, [p]*5)
        # print(np.sum(weights, axis = 1)[:10])
        for j in [0]:
            counts, bins = np.histogram(weights[:, j], bins=50, range=[0.0, 1.0])
            total = sum(counts)
            ax[i].hist(bins[:-1], bins, weights = counts/total,
                label = f'p={p}',  alpha = 0.50,
                #histtype = 'stepfilled',
                histtype = 'step',
                facecolor = colors[i],
                edgecolor = colors[i],
                linewidth = 2)
    
        ax[i].legend()
    
    # from scipy.stats import beta
    # x = np.linspace(0.0, 1.0, 100)
    # for i, (p1,p2) in enumerate(zip(P1,P2)):
    #     ax[-1].plot(x, beta.pdf(x, p1, p2)*1000, color = colors[i], label = f'a,b={p1},{p2}')
    
    #ax[-1].legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')

def plot_beta():
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    gs = fig.add_gridspec(nrows = 1, ncols = 1, hspace=0)
    ax = gs.subplots(sharex=False, sharey=False)
    
    from scipy.stats import beta
    A = [0.1, 0.5, 0.1, 0.5, 0.05]
    B = [0.1, 0.1, 0.5, 0.2, 0.1]
    
    x = np.linspace(0.0, 1.0, 100)
    
    for i, (a,b) in enumerate(zip(A,B)):
        ax.plot(x, beta.pdf(x, a, b), color = colors[i], label = f'a,b={a},{b}')
    
    figname = 'figs/test/beta1.png'
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')
    
    
    
if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-i', '--idx',        action = 'store', dest = 'idx',       default = 1,  type = int, help = 'unique id')
    parser.add_argument('-dp', '--data_path', action = 'store', dest = 'data_path', default = 'data/truth_graph/ttbar',  type = str, help = 'unique id')
    opts = parser.parse_args()
    
    
    #main(opts)
    plot_weights()
    #plot_beta()




