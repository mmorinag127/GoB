

import os, sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/morinaga/.matplotlib/stylelib/nord-light.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*10
colors = np.array(colors)


import torch

from tqdm.auto import tqdm

import utils
from color import nord_color

def query_name(idx):
    if   idx == 0 : return 'pT'
    elif idx == 1 : return 'eta'
    elif idx == 2 : return 'phi'
    elif idx == 3 : return 'Energy'
    elif idx == 4 : return '#of charged'
    elif idx == 5 : return '#of neutral'

def query_range(idx):
    if   idx == 0 : return 40, [0, 400]
    elif idx == 1 : return 50, [-2.5, 2.5]
    elif idx == 2 : return 40, [-3.15, 3.15]
    elif idx == 3 : return 40, [0, 400]
    elif idx == 4 : return 20, [0, 20]
    elif idx == 5 : return 20, [0, 20]

def channel(idx):
    
    if   idx == 0 : return 'track'
    elif idx == 1 : return 'EMCal'
    elif idx == 2 : return 'HCal'
    elif idx == 3 : return 'pT of objects'
    elif idx == 4 : return 'sum of d0'
    elif idx == 5 : return 'charge'
    return None

def ch_range(idx):
    if   idx == 0 : return 10, (0, 10.0)
    elif idx == 1 : return 10, (0, 10.0)
    elif idx == 2 : return 10, (0, 10.0)
    elif idx == 3 : return 100, (0, 50.0)
    elif idx == 4 : return 200, (-10.0, 10.0)
    elif idx == 5 : return 40, (-10.0, 10.0)


def plot_1d(X, bins, range, label, figname, is_norm = False, is_log = False):
    fig = plt.figure(figsize=(8.0, 6.0 ) )
    gs = fig.add_gridspec(nrows = 1, ncols = 1, hspace=0)
    ax = gs.subplots(sharex=False, sharey=False)
    
    
    colors = nord_color.get_colors(N = len(X))
    
    for i, (key, val) in enumerate(X.items()):
        if is_norm:
            total = sum(val['hist'])
            val['hist'] /= total
            
        ax.stairs(val['hist'], val['edge'], 
                    fill = False, 
                    label = key, 
                    alpha = 1.0, 
                    facecolor = colors[i],
                    zorder=0,
                    edgecolor = colors[i], 
                    linewidth = 1.5)
        if is_log:
            ax.set_yscale('log')
    ax.set_xlim(range)
    #ax.minorticks_on()
    plt.xlim(*range)
    
    ax.set_xlabel(label, fontsize = 20)
    ax.set_ylabel('count', fontsize = 20)
    ax.grid()
    ax.legend(frameon=False)
    
    plt.tight_layout()
    figname = figname.replace(' ', '_')
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'{figname} is done...')


def mkdir(dir):
    import os
    if not os.path.exists(f'{dir}'):
        os.makedirs(f'{dir}')

class Hist:
    def __init__(self, name, bins, range, device):
        self.name = name
        self.bins = bins
        self.range = range
        hist, edge = torch.histogram(torch.zeros(0), bins = self.bins, range = self.range)
        self.hist = hist.to(device)
        self.edge = edge.to(device)
        
    def add(self, image):
        image = image.flatten()
        mask = image != 0.0
        image = image[mask]
        h, e = torch.histogram(image, bins = self.bins, range = self.range)
        self.hist += h

class Average:
    def __init__(self, sample, n_channel, n_pixel, n_query, device, pt_cut = -1.0 ):
        self.n_channel = n_channel
        self.n_query = n_query
        self.sample = sample
        self.pt_cut = pt_cut
        self.hists = []
        for idx in range(n_channel):
            bins, bin_range = ch_range(idx)
            self.hists.append(Hist(name = channel(idx), bins = bins, range = bin_range, device = device))
        
        
        self.query = []
        for idx in range(n_query):
            bins, bin_range = query_range(idx)
            self.query.append(Hist(name = query_name(idx), bins = bins, range = bin_range, device = device))
        
    def add(self, image, query):
        if self.pt_cut > 0 :
            mask = query[:, 0] > self.pt_cut
            image = image[mask]
            query = query[mask]
        
        
        for idx in range(self.n_channel):
            self.hists[idx].add(image[:, idx, :, :])
        
        for idx in range(self.n_query):
            self.query[idx].add(query[:, idx])
    
def calc_avg_image(data_path, n_events, flavors, n_channel = 5, n_pixel = 32, dR = 0.4, data_type = 'reco', apply_pt_scale = True, norm_jet_pt = True, eps = 1.0e-7) :
    n_query = 6
    #device = torch.device('cuda:2')
    device = torch.device('cpu')
    avg = {}
    #avg['all'] = Average('all', n_channel, n_pixel, n_query = n_query, device = device, pt_cut = 100.0)
    
    sys.path.append('src')
    
    from h5py_dataset import make_flavor_dataset
    for iflavor, flavor in enumerate(flavors):
        
        avg[flavor] = Average(flavor, n_channel, n_pixel, n_query = 6, device = device, pt_cut = 100.0)
        
        dataset = make_flavor_dataset(
            data_path, [flavor], {flavor:0}, n_samples = {flavor: n_events}, 
            n_sample_base = 2**20,
            n_pixel = n_pixel, 
            dR = dR, 
            n_channel = n_channel,
            n_query = 6, 
            data_type = data_type,
            apply_pt_scale = apply_pt_scale,
            norm_jet_pt = norm_jet_pt,
            image_transforms = None,
            query_transforms = None,
            from_array = False,
            )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size = 256, num_workers = 32, pin_memory = True, drop_last = False)
        
        pbar_args = dict(total = len(dataloader), unit = ' batch', ncols = 250, disable = False, bar_format="{desc}{percentage:3.0f}%| {n_fmt: >4}/{total_fmt: >4} [{rate_fmt: >16}{postfix}]",)
        pbar_desc = f'[{flavor} {iflavor+1}/{len(flavors)}]'
        
        with tqdm(**pbar_args) as pbar:
            pbar.set_description(pbar_desc)
            for idx, (image, query, label) in enumerate(dataloader):
                image = image.to(device)
                query = query.to(device)
                
                avg[flavor].add(image, query)
                
                pbar.update(1)
        
    return avg

def main(opts, fig_dir, data_type, n_pixel = 32, n_channel = 6):
    
    flavors = ['d-jet', 'u-jet', 's-jet', 'c-jet', 'b-jet', 'g-jet']
    # flavors += ['gd-jet', 'gu-jet', 'gs-jet', 'gc-jet', 'gb-jet']
    
    avg = calc_avg_image(data_path = f'data/{opts.data_ver}', n_events = opts.n_events, 
                        flavors = flavors, n_channel = n_channel, n_pixel = n_pixel, 
                        data_type = opts.data_type, 
                        )
    
    mkdir(fig_dir)
    for idx in range(n_channel):
        channel_name = channel(idx)
        bins, bin_range = ch_range(idx)
        hists = {}
        for flavor in flavors:
            hists[flavor] = {}
            hists[flavor]['hist'] = avg[flavor].hists[idx].hist.cpu().numpy()
            hists[flavor]['edge'] = avg[flavor].hists[idx].edge.cpu().numpy()
            
        figname = f'{fig_dir}/{channel_name}.png'
        plot_1d(hists, bins = bins, range = bin_range, label = channel_name, figname = figname, is_log = False, is_norm = False)
        figname = f'{fig_dir}/{channel_name}_log.png'
        plot_1d(hists, bins = bins, range = bin_range, label = channel_name, figname = figname, is_log = True, is_norm = False)

        figname = f'{fig_dir}/{channel_name}_norm.png'
        plot_1d(hists, bins = bins, range = bin_range, label = channel_name, figname = figname, is_log = False, is_norm = True)
        figname = f'{fig_dir}/{channel_name}_norm_log.png'
        plot_1d(hists, bins = bins, range = bin_range, label = channel_name, figname = figname, is_log = True, is_norm = True)
    
    
    n_query = 6
    for idx in range(n_query):
        channel_name = query_name(idx)
        bins, bin_range = query_range(idx)
        hists = {}
        for flavor in flavors:
            hists[flavor] = {}
            hists[flavor]['hist'] = avg[flavor].query[idx].hist.cpu().numpy()
            hists[flavor]['edge'] = avg[flavor].query[idx].edge.cpu().numpy()
            
        figname = f'{fig_dir}/{channel_name}.png'
        plot_1d(hists, bins = bins, range = bin_range, label = channel_name, figname = figname, is_log = False, is_norm = False)
        figname = f'{fig_dir}/{channel_name}_log.png'
        plot_1d(hists, bins = bins, range = bin_range, label = channel_name, figname = figname, is_log = True, is_norm = False)

        figname = f'{fig_dir}/{channel_name}_norm.png'
        plot_1d(hists, bins = bins, range = bin_range, label = channel_name, figname = figname, is_log = False, is_norm = True)
        figname = f'{fig_dir}/{channel_name}_norm_log.png'
        plot_1d(hists, bins = bins, range = bin_range, label = channel_name, figname = figname, is_log = True, is_norm = True)
        
        




if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-ne', '--n_events',   action = 'store', dest = 'n_events',   default = 9999,     type = int)
    parser.add_argument('-dv', '--data_ver',   action = 'store', dest = 'data_ver',  default = '2022-02-09',  type = str)
    parser.add_argument('-dt', '--data_type',  action = 'store', dest = 'data_type', default = 'reco', type = str)
    opts = parser.parse_args()
        
    
    fig_dir = f'figs/jet-property-v2-pt100/{opts.data_ver}'
    n_channel = 6
    
    main(opts, fig_dir, data_type = opts.data_type, n_pixel = 32, n_channel = n_channel)
    
    



















