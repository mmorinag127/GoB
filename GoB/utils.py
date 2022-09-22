import jax
import jax.numpy as jnp
import jmp
import optax
import numpy as np
from copy import deepcopy

def set_seed(seed = 1, logger = None):
    if logger is not None : 
        logger.info(f'seed is fixed to {seed}')
    import os
    import random
    import numpy as np
    import tensorflow as tf
    from tensorflow.random import set_seed
    tf.random.set_seed(seed)
    #import torch
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def make_mixed_precision(config):
    if config.setup.mixed_precision:
        def initial_loss_scale():
            return jmp.DynamicLossScale(jmp.half_dtype()(config.mp_policy.init_val))
        
        def make_mp_policy():
            return jmp.Policy(param_dtype = jnp.float32, compute_dtype = jnp.float16, output_dtype = jnp.float32)
    
    else:
        def initial_loss_scale():
            return jmp.NoOpLossScale()
        
        def make_mp_policy():
            return jmp.Policy(param_dtype = jnp.float32, compute_dtype = jnp.float32, output_dtype = jnp.float32)
        
    return make_mp_policy, initial_loss_scale

def make_optimizer(config, n_step_per_epoch):
    opts = []
    for key, val in config.optimizer.opts.items():
        if hasattr(optax, key):
            if val is not None:
                opts += [getattr(optax, key)(**val)]
            else:
                opts += [getattr(optax, key)()]
        else:
            raise ValueError(f'make_optimizer dose not suppprt {key}!!')
    
    lr_scheduler = make_lr_scheduler(config, n_step_per_epoch)
    opt_name = config.setup.optimizer
    if hasattr(optax, opt_name):
        kwargs = deepcopy(config.optimizer[opt_name])
        kwargs['learning_rate'] = lr_scheduler
        opts.append(getattr(optax, opt_name)(**kwargs))
    else:
        raise ValueError(f'{opt_name} is not known...')
    
    def _make_optimizer():
        return optax.chain(*opts)
    
    return _make_optimizer

def make_lr_scheduler(config, n_step_per_epoch):
    
    lr_name = config.setup.lr_scheduler 
    if hasattr(optax, lr_name):
        kwargs = {}
        for key, val in config.lr_scheduler[lr_name].items():
            if 'steps' in key:
                kwargs[key] = val * n_step_per_epoch
            else:
                kwargs[key] = val
            
        lr_scheduler = getattr(optax, lr_name)(**kwargs)
    else:
        raise ValueError(f'make_lr_scheduler dose not suppprt {lr_name}!!')
    
    return lr_scheduler

def make_loss(config): 
    
    if 'cb_ce_loss' in config.setup.loss:
        from loss import make_cb_ce_loss
        loss_f = make_cb_ce_loss(config)
    
    elif 'cb_bce_loss' in config.setup.loss:
        from loss import make_cb_bce_loss
        loss_f = make_cb_bce_loss(config)
    
    elif 'ce_loss' in config.setup.loss:
        from loss import make_softmax_ce_loss
        loss_f = make_softmax_ce_loss(config)
        
    else:
        raise ValueError(f'There is no {config.setup.loss}!!')
    
    return loss_f

def calc_accuracy(logits, one_hot_labels, mask = None):
    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0]])
    error_rate = (((jnp.argmax(logits, -1) != jnp.argmax(one_hot_labels, -1))) * mask).sum() / mask.sum()
    # Set to zero if there is no non-masked samples.
    return jnp.nan_to_num(error_rate)

def calc_accuracy_topk(logits, one_hot_labels, k = 5, mask = None):
    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0]])
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    top_k_preds = jnp.argsort(logits, axis=-1)[:, -k:]
    hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
    error_rate = 1 - ((hit * mask).sum() / mask.sum())
    # Set to zero if there is no non-masked samples.
    return jnp.nan_to_num(error_rate)

# from jax.experimental.host_callback import id_print

def topK_acc(logits, one_hot_labels, K = [1,5]):
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    aaa = jnp.argsort(logits, axis=-1)
    accs = []
    for k in K:
        preds = aaa[:, -k:]
        hit = jax.vmap(jnp.isin)(true_labels, preds)
        acc = hit.sum()
        acc = jax.device_get(acc) 
        accs.append(acc)
    
    return accs

import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.gpu_usage = {}
        self.total = 0

    def run(self):
        while not self.stopped:
            for i, res in enumerate(GPUtil.getGPUs()):
                if f'GPU{i}' in self.gpu_usage.keys():
                    self.gpu_usage[f'GPU{i}'] += res.load*100.0
                else:
                    self.gpu_usage[f'GPU{i}'] = 0
            time.sleep(self.delay)
            self.total += 1

    def stop_record(self):
        self.stopped = True
    
    def start_record(self):
        self.start()
    
    def clear_usage(self):
        self.gpu_usage = {}
        self.total = 0
    
    def get_gpu_usage(self, mean = False):
        if self.total > 0:
            _gpu_usage = { f'GPU/{k}': v/self.total for k,v in self.gpu_usage.items()}
        else:
            print('gpu_usage is all zero!!!!')
            _gpu_usage = { f'GPU/{k}': 0 for k,v in self.gpu_usage.items()}
        
        if mean:
            _gpu_usage = {'GPU': np.mean([v for v in _gpu_usage.values() if v > 0.0]) }
        
        return _gpu_usage
    





