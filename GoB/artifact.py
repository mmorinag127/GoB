
from contextlib import contextmanager
import time, datetime
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import math
#import tensorboardX as tbX


from logger import make_logger
logger = make_logger(__name__)
import utils

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
        self.metrics_p = list(config.metrics_print)
        self.n_epochs = config.setup.n_epochs
        self.info_step = config.setup.n_data.info_step
        self.name = config.setup.model_name
        self.workdir = config.workdir
        self.is_plot = config.is_plot
        self.process_idx = config.process_idx
        disable_tqdm = config.disable_tqdm
        if self.process_idx != 0:
            disable_tqdm = True
        
        self.data_length = data_length
        l = len(f'{self.data_length}')
        bar_format="{desc}{percentage:3.0f}% {n_fmt: >"+f'{l}'+"}/{total_fmt: >" +f'{l}'+ "} [{rate_fmt: >8}{postfix}]"
        self.pbar_args = dict(total = data_length + 1, unit = 'b', ncols = 250, disable = disable_tqdm, bar_format=bar_format,)
        self.pbar_desc = lambda epoch: f'{time.strftime("%H:%M:%S")}: [{epoch: >3}/{self.n_epochs}]'
        
        self.history = {phase:{key:[]  for key in self.metrics+['time']} for phase in self.phases}
        self.values  = {phase:{key:0.0 for key in self.metrics+['time']} for phase in self.phases}
        self.total   = {phase: 0.0 for phase in self.phases}
        self.monitor = {phase: {} for phase in self.phases}
        
        self.best_loss = float(jnp.inf)
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
        self.hparams['model'] = deepcopy(config.model[config.setup.model])
        
        self.hparams = flatten_nested_dict(self.hparams)
        self.step = 0
    
    def close(self):
        pass
    
    @contextmanager
    def timer(self, phase, epoch):
        start = time.time()
        yield
        end = time.time()
        self.monitor[phase]['time'] = end - start
    
    def set_metrics(self, phase, metrics):
        self.total[phase] += metrics['total']
        for key in self.metrics:
            if key in metrics.keys():
                self.values[phase][key] += jnp.sum(metrics[key])
    
    def set_monitor(self, phase, metrics):
        for key, value in metrics.items():
            self.monitor[phase][key] = value
    
    def check_best_epoch(self, epoch):
        last_loss = self.values['test']['loss']
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
            for metric in self.metrics:
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
            
        
        self.count = 0
        self.step = 0
        return is_best
    
    def get_metrics(self):
        metrics = {}
        for phase in self.phases:
            for metric in self.metrics:
                metrics[f'{metric}/{phase}'] = self.history[phase][metric][-1][0]
        return metrics
    
    def set_pbar_postfix_str(self, is_best = None):
        postfix = ''
        for metric in self.metrics_p:
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
                    elif 'norm' in metric:
                        r = f'{val: .3e}'
                    else:
                        r = f'{val: .3f}'
                
                ret.append(color(phase)  + r + reset)
            
            if len(ret) > 0 :
                if 'norm' not in metric:
                    ret = '/'.join(r.replace(' ', '') for r in ret)
                else:
                    ret = ret[0].replace(' ', '')
                postfix += f'{metric}:{ret}, '
        
        for metric in self.monitor[phase].keys():
            ret = []
            for phase in self.phases:
                if self.monitor[phase][metric] is not None and self.monitor[phase][metric] > 0.0:
                    r = f'{self.monitor[phase][metric]: .0f}'
                    ret.append(color(phase)  + r + reset)
            if len(ret) > 0 :
                if 'norm' not in metric:
                    ret = '/'.join(r.replace(' ', '') for r in ret)
                else:
                    ret = ret[0].replace(' ', '')
                postfix += f'{metric}:{ret}, '
        #postfix += f'step:{self.step:>3d}'
        self.step += 1
        if is_best is not None and is_best:
            postfix += red + 'Best' + reset
        else:
            postfix = postfix[:-2] # to cut out ', ' at the end of postfix
        
        postfix = postfix.replace(': ', ':').replace('/ ', '/').replace('  ', ' ')
        
        self.postfix = postfix
    
    def pbar_update(self, is_best = None):
        if self.count % self.info_step == 0 or self.count == 0 or self.count == self.data_length:
            self.set_pbar_postfix_str(is_best)
        
        self.count += 1
        return self.postfix
    
    def save_checkpoint(self, run_state, name = 'checkpoint_best.pkl'):
        import pickle
        checkpoint_name = f'{self.workdir}/{name}'
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
    
    def save_outputs(self, name, xlists = ['logit', 'label']):
        dic = {}
        for key in xlists:
            x = np.concatenate(self.outputs[key], axis = 0)
            dic[key] = x
            print(f'{key} : {x.shape}')
        
        file = f'{self.workdir}/{name}.npz'
        
        if 'expert_weights' in self.outputs.keys():
            expert_weights = np.concatenate(self.outputs['expert_weights'], axis = 0)
            print(f'expert : {expert_weights.shape}')
            dic['expert_weights'] = expert_weights
        np.savez_compressed(file, **dic)
        logger.info(f'{file} is saved.')

class Artifact2:
    def __init__(self, config):
        self.phases = config.phases
        self.metrics_p = list(config.metrics_print)
        self.aux_to_metric = list(config.aux_to_metric)
        self.metrics = list(config.metrics) + self.aux_to_metric
        self.n_step = int(config.setup.n_data.n_step.train)
        self.info_step = config.info_step
        self.name = config.setup.model_name
        self.workdir = config.workdir
        self.is_plot = config.is_plot
        self.process_idx = config.process_idx
        disable_tqdm = config.disable_tqdm if self.process_idx == 0 else True
        
        l = math.ceil(math.log10(self.n_step))
        t = red + '{remaining}' + reset + '/' + blue + '{elapsed}' + reset
        bar_format="{desc}{percentage:3.0f}% {n_fmt: >"+f'{l}'+"}/{total_fmt: >" +f'{l}'+ "} [{rate_fmt: >8} " + t + "{postfix}]"
        self.pbar_args = dict(total = self.n_step, unit = 'b', ncols = 250, disable = disable_tqdm, bar_format=bar_format,)
        self.pbar_desc = lambda : f'{time.strftime("%H:%M:%S")}'
        
        self.history = {phase:{key:[]  for key in self.metrics+['time']} for phase in self.phases}
        self.values  = {phase:{key:0.0 for key in self.metrics+['time']} for phase in self.phases}
        self.total   = {phase: 0.0 for phase in self.phases}
        self.monitor = {phase: {} for phase in self.phases}
        
        self.best_loss = float(jnp.inf)
        self.best_epoch = None
        self.count = 0
        self.postfix = ''
        
        self.history['epoch'] = []
        self.outputs = {}
        self.hparams = {}
        self.is_log_param = config.setup.log_param
        
        for key, val in config.hparams.items():
            if type(val) is list:
                for v in val:
                    k = config.setup[v]
                    self.hparams[v] = k
                    self.hparams[key] = deepcopy(config.config[v][k])
            else:
                self.hparams[key] = deepcopy(val)
        # self.hparams['model'] = deepcopy(config.model[config.setup.model])
        
        self.hparams = flatten_nested_dict(self.hparams)
        self.step = 0
        
        self.phistory = {}
        self.nan_fill = True
        for key in ['param', 'grad']:
            self.phistory[key] = ParameteeTracker(name = f'{self.workdir}/{key}')
        
    def update_hparams(self, hp):
        self.hparams.update(hp)
        self.hparams = flatten_nested_dict(self.hparams)
    
    def close(self):
        pass
    
    @contextmanager
    def timer(self, phase, epoch):
        start = time.time()
        yield
        end = time.time()
        self.monitor[phase]['time'] = end - start
    
    def add_metrics(self, phase, metrics):
        self.total[phase] += metrics['total']
        for key in self.metrics:
            if key in metrics.keys():
                self.values[phase][key] += np.sum(metrics[key])
    
    def calc_metrics(self):
        vals = {}
        for metric in self.metrics_p:
            for phase in self.phases:
                val = self.values[phase][metric]
                if self.total[phase] > 0 and metric in self.values[phase].keys():
                    val = val / self.total[phase]
                vals[metric][phase] = val
        return vals
    
    def set_metrics(self, phase, metrics):
        for key in self.metrics:
            if key in metrics.keys():
                self.values[phase][key] = metrics[key]
    
    def set_monitor(self, phase, metrics):
        for key, value in metrics.items():
            self.monitor[phase][key] = value
    
    def check_best_epoch(self, epoch):
        last_loss = self.values['test']['loss']
        is_best = False
        
        if last_loss < self.best_loss:
            self.best_loss = last_loss
            self.best_epoch = epoch
            is_best = True
        return is_best
    
    def end_of_epoch(self, epoch):
        is_best = self.check_best_epoch(epoch) if 'test' in self.phases else False
        self.history['epoch'].append(epoch)
        
        for phase in self.phases:
            for metric in self.metrics:
                val = self.values[phase][metric]
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
                # for phase in ['train', 'test']:
                #     for key,val in history[phase].items():
                #         print(f'{phase} {key} : {type(val[0][0])} {val[0][0]}')
                json.dump([history], f, indent = 2, cls=NumpyEncoder)
                
            if self.is_plot:
                from plot_history import plot_history
                plot_history(self.workdir, history)
        
        self.count = 0
        self.step = 0
        return is_best
    
    def get_metrics(self):
        metrics = {}
        for phase in self.phases:
            for metric in self.metrics:
                metrics[f'{metric}/{phase}'] = self.history[phase][metric][-1][0]
        return metrics
    
    def set_pbar_postfix_str(self, is_best = None, istep = None):
        postfix = ''
        if istep is not None:
            postfix += f'istep:{istep} '
        
        for metric in self.metrics_p:
            ret = []
            for phase in self.phases:
                val = self.values[phase][metric]
                if val == 0.0:
                    r = f'None'
                else:
                    if 'loss' in metric:
                        r = f'{val: .3e}'
                    elif 'acc' in metric:
                        r = f'{val: .2f}'
                        r = f'{r: >6s}'
                    elif 'lr' in metric:
                        r = f'{val: .3e}'
                    elif 'norm' in metric:
                        r = f'{val: .3e}'
                    else:
                        r = f'{val: .3f}'
                
                ret.append(color(phase)  + r + reset)
            
            if len(ret) > 0 :
                if 'norm' in metric or 'lr' in metric:
                    ret = ret[0].replace(' ', '')
                elif 'acc' in metric:
                    ret = ret[1].replace(' ', '')
                else:
                    ret = '/'.join(r.replace(' ', '') for r in ret)
                    
                postfix += f'{metric}:{ret}, '
        
        for metric in self.monitor[phase].keys():
            ret = []
            for phase in self.phases:
                if self.monitor[phase][metric] is not None and self.monitor[phase][metric] > 0.0:
                    r = f'{self.monitor[phase][metric]: .0f}'
                    ret.append(color(phase)  + r + reset)
            if len(ret) > 0 :
                if 'norm' not in metric:
                    ret = '/'.join(r.replace(' ', '') for r in ret)
                else:
                    ret = ret[0].replace(' ', '')
                postfix += f'{metric}:{ret}, '
        #postfix += f'step:{self.step:>3d}'
        self.step += 1
        if is_best is not None and is_best:
            postfix += red + 'Best' + reset
        else:
            postfix = postfix[:-2] # to cut out ', ' at the end of postfix
        
        postfix = postfix.replace(': ', ':').replace('/ ', '/').replace('  ', ' ')
        self.postfix = postfix
    
    def pbar_update(self, is_best = None, istep = None):
        #if self.count % self.info_step == 0 or self.count == 0 or self.count == self.data_length:
        #if self.count % self.info_step == 0 or self.count == 0:
        self.set_pbar_postfix_str(is_best, istep = istep)
        
        self.count += 1
        return self.postfix
    
    def save_checkpoint(self, run_state, name = 'checkpoint_best.pkl'):
        import pickle
        checkpoint_name = f'{self.workdir}/{name}'
        with open(checkpoint_name, 'wb') as f:
            pickle.dump(run_state, f)
        return checkpoint_name
    
    def add_x(self, x, name, shape = None):
        if name not in self.outputs.keys():
            self.outputs[name] = []
        # print(name, x.shape)
        # if shape is None:
        #     shape = [-1, x.shape[-1]]
        #     print(shape)
        # xx = jnp.reshape(x, shape)
        # print(xx.shape)
        self.outputs[name].append(x)
    
    def save_outputs(self, name, xlists = ['logit', 'label'], label_table = None, label_name = None):
        dic = {}
        for key in xlists:
            x = np.concatenate(self.outputs[key], axis = 0)
            dic[key] = x
            print(f'{key} to be saved : {x.shape}')
        
        file = f'{self.workdir}/{name}.npz'
        if label_table is not None:
            dic['label_table'] = label_table
        if label_name is not None:
            dic['label_name'] = label_name
        
        np.savez_compressed(file, **dic)
        logger.info(f'{file} is saved.')
    
    def log_params_nan(self, step, **kwargs):
        if not self.is_log_param:
            return 
        
        if self.nan_fill :
            for key, val in kwargs.items():
                self.phistory[key].add_nan(step, val)
            self.nan_fill = False
        
    def log_params(self, step, **kwargs):
        if not self.is_log_param:
            return 
        for key, val in kwargs.items():
            self.phistory[key].add(val, step = step)
            

class ParameteeTracker:
    def __init__(self, name):
        self.name = name
        self.current = -1
        self.steps = []
        utils.mkdir(self.name)
    
    def add(self, params, step):
        self.steps.append(step)
        self.current += 1
        flatten = utils.param_flatten(params, is_root=True)
        name = f'{self.name}/{self.current:08d}.npz'
        np.savez_compressed(name, **flatten)
        
    def add_nan(self, step, params):
        flatten = utils.param_flatten(params, is_root=True)
        name = f'{self.name}/nan_at_{step}.npz'
        np.savez_compressed(name, **flatten)
        

import json
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)














