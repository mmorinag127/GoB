
from contextlib import contextmanager
import time, datetime
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

import tensorboardX as tbX


from logger import make_logger
logger = make_logger(__name__)


grey, yellow, green, blue, red, reset = "\x1b[38;20m", "\x1b[33;20m", "\x1b[32;20m", "\x1b[34;20m", "\x1b[31;20m", "\x1b[0m"
def color(phase):
    if 'train' in phase:
        return blue
    elif 'test' in phase:
        return yellow
    elif 'valid' in phase:
        return green
    return grey

def flatten_nested_dict(nested, separator = '/'):
    stack = list(nested.items())
    ans = {}
    while stack:
        key, val = stack.pop()
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                stack.append((f"{key}{separator}{sub_key}", sub_val))
        else:
            ans[key] = val
    return ans

class Artifact:
    def __init__(self, config, data_length):
        self.phases = config.phases
        self.metrics = list(config.metrics)
        self.n_epochs = config.setup.n_epochs
        self.info_step = config.info_step
        self.name = config.setup.model_name
        self.workdir = config.workdir
        self.is_plot = config.is_plot
        self.process_idx = config.process_idx
        disable_tqdm = config.disable_tqdm
        if self.process_idx != 0:
            disable_tqdm = True
        
        self.data_length = data_length
        self.pbar_args = dict(total = data_length + 1, unit = ' batch', ncols = 250, disable = disable_tqdm, 
                                bar_format="{desc}{percentage:3.0f}% {n_fmt: >5}/{total_fmt: >5} [{rate_fmt: >16}{postfix}]",)
        self.pbar_desc = lambda epoch: f'{time.strftime("%H:%M:%S")}: [{epoch: >4}/{self.n_epochs}]'
        
        self.history = {phase:{key:[]  for key in self.metrics+['time']} for phase in self.phases}
        self.values  = {phase:{key:0.0 for key in self.metrics+['time']} for phase in self.phases}
        self.total   = {phase: 0.0 for phase in self.phases}
        self.monitor = {phase: {} for phase in self.phases}
        
        self.best_loss = None
        self.best_epoch = None
        self.count = 0
        self.postfix = ''
        
        self.history['epoch'] = []
        self.outputs = {}
        self.hparams = {}
        
        for key, val in config.hparams.items():
            if type(val) is list:
                for v in val:
                    k = config.setup[v]
                    self.hparams[v] = k
                    self.hparams[key] = deepcopy(config.config[v][k])
                    
            else:
                self.hparams[key] = deepcopy(val)
        self.hparams = flatten_nested_dict(self.hparams)
        
        if not config.disable_tb:
            current_time = datetime.datetime.now().strftime("%m%d-%H:%M")
            self.summary_writer = {}
            for phase in self.phases:
                self.summary_writer[phase] = tbX.SummaryWriter(f'{self.workdir}/tbX/{current_time}/{phase}')
        else:
            self.summary_writer = None
    
    def close(self):
        if self.summary_writer is not None:
            for phase in self.phases:
                self.summary_writer[phase].close()
    
    @contextmanager
    def timer(self, phase, epoch):
        start = time.time()
        yield
        end = time.time()
        self.values[phase]['time'] = end - start
    
    def set_metrics(self, phase, metrics):
        self.total[phase] += metrics['total']
        for key in self.metrics:
            if key in metrics.keys():
                self.values[phase][key] += jnp.sum(metrics[key])
    
    def set_monitor(self, phase, metrics):
        for key, value in metrics.items():
            if key not in self.monitor[phase].keys():
                self.monitor[phase][key] = [value]
            else:
                self.monitor[phase][key] += [value]
    
    def check_best_epoch(self, epoch):
        last_loss = self.values['test']['loss']
        if self.best_epoch is None:
            self.best_loss = last_loss
            self.best_epoch = epoch
        
        is_best = False
        
        if last_loss < self.best_loss:
            self.best_loss = last_loss
            self.best_epoch = epoch
            is_best = True
        return is_best
    
    def end_of_epoch(self, epoch, params = None):
        is_best = self.check_best_epoch(epoch) if 'test' in self.phases else False
        self.set_pbar_postfix_str(is_best)
        self.history['epoch'].append(epoch)
        
        for phase in self.phases:
            for metric in self.metrics + ['time']:
                if self.total[phase] > 0 and metric in self.values[phase].keys():
                    val = jax.device_get(self.values[phase][metric]/self.total[phase] )
                    val = val.item()
                else:
                    val = 0.0
                self.history[phase][metric].append( [val] )
                
                self.values[phase][metric] = 0.0
            self.total[phase] = 0.0
            
            for key, value in self.monitor[phase].items():
                if key not in self.history[phase].keys():
                    self.history[phase][key] = [value]
                else:
                    self.history[phase][key] += [value]
        
        if self.process_idx == 0:
            with open(f'{self.workdir}/history-{self.name}.json', 'w') as f:
                import json
                history = self.history
                history['best_epoch'] = self.best_epoch
                history['epoch'] = self.history['epoch']
                json.dump([history], f, indent = 2)
                
            if self.is_plot:
                from plot_history import plot_history
                plot_history(self.workdir, history)
            
            if self.summary_writer is not None:
                for phase in self.phases:
                    metrics = {}
                    for metric in self.metrics:
                        self.summary_writer[phase].add_scalar(metric, history[phase][metric][-1][0], global_step = epoch)
                        metrics[metric] = history[phase][metric][-1][0]
                        
                    
                    for key, val in self.monitor[phase].items():
                        self.summary_writer[phase].add_scalar(key, val[-1], global_step = epoch)
                        
                    self.summary_writer[phase].add_hparams(self.hparams, metrics)
                    
                    if params is not None:
                        self.add_param(phase, params, key = 'model')
                    self.summary_writer[phase].flush()
        
        self.count = 0
        return is_best
    
    def get_metrics(self):
        metrics = {}
        for phase in self.phases:
            for metric in self.metrics:
                metrics[f'{metric}/{phase}'] = self.history[phase][metric][-1][0]
        return metrics
    
    def add_param(self, phase, param, key):
        if type(param) is not dict:
            self.summary_writer[phase].add_histogram(key, jax.device_get(param))
        else:
            for k, p in param.items():
                self.add_param(phase, p, f'{key}/{k}' if len(key) > 0 else f'{k}')
    
    def set_pbar_postfix_str(self, is_best = None):
        postfix = ''
        for metric in self.metrics:
            ret = []
            for phase in self.phases:
                val = self.values[phase][metric]
                if self.total[phase] > 0 and metric in self.values[phase].keys():
                    val = val / self.total[phase]
                if val == 0.0:
                    r = f'0.0'
                else:
                    if 'loss' in metric:
                        r = f'{val: .3e}'
                    elif 'acc' in metric:
                        r = f'{val: .2f}'
                        r = f'{r: >6s}'
                    elif 'lr' in metric:
                        r = f'{val: .3e}'
                    else:
                        r = f'{val: .3f}'
                
                ret.append(color(phase)  + r + reset)
            
            if len(ret) > 0 :
                ret = '/'.join(r for r in ret)
                postfix += f'{metric}:{ret}, '
        ret = []
        for phase in self.phases:
            if self.values[phase]['time'] is not None and self.values[phase]['time'] > 0.0:
                r = f'{self.values[phase]["time"]: .2f}'
                #r = f'{r: >8s}'
                ret.append(color(phase)  + r + reset)
        if len(ret) > 0 :
            ret = '/'.join(r for r in ret)
            postfix += f'time:{ret}, '
        
        if is_best is not None and is_best:
            postfix += red + ' Best' + reset
        self.postfix = postfix
    
    def pbar_update(self, is_best = None):
        if self.count % self.info_step == 0 or self.count == 0 or self.count == self.data_length:
            self.set_pbar_postfix_str(is_best)
        
        self.count += 1
        return self.postfix
    
    def save_checkpoint(self, run_state):
        import pickle
        checkpoint_name = f'{self.workdir}/checkpoint_best.pkl'
        with open(checkpoint_name, 'wb') as f:
            pickle.dump(run_state, f)
        return checkpoint_name
    
    def add_x(self, x, name, shape = None):
        if name not in self.outputs.keys():
            self.outputs[name] = []
        
        if shape is None:
            shape = [-1, x.shape[-1]]
        xx = jnp.reshape(x, shape)
        
        self.outputs[name].append(xx)
    
    def save_outputs(self, name):
        logits = np.concatenate(self.outputs['logit'], axis = 0)
        labels = np.concatenate(self.outputs['label'], axis = 0)
        props  = np.concatenate(self.outputs['prop'],  axis = 0)
        file = f'{self.workdir}/{name}.npz'
        np.savez_compressed(file, labels = labels, logits = logits, props = props)
        logger.info(f'{file} is saved.')
    
















