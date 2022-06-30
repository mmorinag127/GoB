from difflib import restore
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#os.environ["XLA_FLAGS"]="--xla_cpu_enable_xprof_traceme"
#os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=32'


import tensorflow as tf
#tf.config.experimental.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'TF: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.get_logger().setLevel('ERROR')


import argparse


import jax
import jax.numpy as jnp
jnp.set_printoptions(threshold=2)
jax.config.update("jax_debug_nans", True)


from tqdm.auto import tqdm

import utils
from config import Config
from model.utils import make_model, make_model_info
from dataset import dataset
from artifact import Artifact
#from updater import Updater

from typing import NamedTuple
import functools
import optax
import jmp
import haiku as hk
from copy import deepcopy

from logger import make_logger
logger = make_logger(__name__)

class RunState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    loss_scale: jmp.LossScale
    rng: jax.random.PRNGKey

def main(opts):
    
    config = Config(opts, **{'setup:model': opts.model, 'setup:seed': opts.seed})
    make_mp_policy, initial_loss_scale = utils.make_mixed_precision(config)
    
    mp_policy = make_mp_policy()
    forward = make_model(config, mp_policy)

    local_device_count = jax.local_device_count()
    data_info, datasets = {}, {}
    
    for phase in config.phases:
        data_info[phase], datasets[phase] = dataset.make_dataset(phase = phase, **config.dataset, dtype = mp_policy.compute_dtype)
        #data_info[phase]['n_step_per_epoch'] = 20
    
    if config.mode.training():
        data_length = data_info['train']['n_step_per_epoch'] + data_info['test']['n_step_per_epoch']
    elif config.mode.eval():
        data_length = data_info['test']['n_step_per_epoch']
    
    artifact = Artifact(config, metrics = ['loss','top1_acc','top2_acc'], 
                        data_length = data_length,
                        process_idx = jax.process_index(),
                        is_plot = False,
                        )
    
    optimizer = utils.make_optimizer(config, data_info['train']['n_step_per_epoch'])()
    loss_f = utils.make_loss(config)
    
    #updater = Updater(forward, loss_f, optimizer, mp_policy, config.mp_nonfinite, initial_loss_scale)
    
    def initial_state(rng, data):
        out_rng, init_rng =  jax.random.split(rng)
        params, state = forward.init(init_rng, data, training = True)
        opt_state = optimizer.init(params)
        loss_scale = initial_loss_scale()
        return RunState(params, state, opt_state, loss_scale, out_rng)
    
    def restore_from_file(rng):
        import pickle
        file = f'{config.workdir}/checkpoint_best.pkl'
        with open(file, 'rb') as f:
            run_state = pickle.load(f)
        params, state, opt_state, loss_scale, _ = run_state
        return RunState(params, state, opt_state, loss_scale, rng)
    
    def loss_step(params, state, rng, loss_scale, batch):
        logits, state = forward.apply(params, state, rng, batch, training = True)
        loss = loss_f(logits, batch['label'], params)
        return loss_scale.scale(loss), (loss, state)
    
    @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
    def update(run_state, batch):
        params, state, opt_state, loss_scale, rng = run_state 
        
        rng, new_rng = jax.random.split(rng)
        grads, (_, new_state) = (jax.grad(loss_step, has_aux = True)(params, state, rng, loss_scale, batch))
        
        grads = mp_policy.cast_to_compute(grads)
        grads = loss_scale.unscale(grads)
        
        grads = jax.lax.pmean(grads, axis_name='batch')
        grads = mp_policy.cast_to_param(grads)
        
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        if config.mp_nonfinite:
            grads_finite = jmp.all_finite(grads)
            loss_scale = loss_scale.adjust(grads_finite)
            new_params, new_state, new_opt_state = jmp.select_tree(grads_finite, (new_params, new_state, new_opt_state), (params, state, opt_state))
        
        state = jmp.cast_to_full(state)
        return RunState(new_params, new_state, new_opt_state, loss_scale, new_rng)
    
    @functools.partial(jax.pmap, axis_name='batch')
    def eval_step(run_state, batch):
        params, state, _, _, rng = run_state
        rng, new_rng = jax.random.split(rng)
        logits, _ = forward.apply(params, state, new_rng, batch, training = False)
        one_hot_labels = jax.nn.one_hot(batch['label'], config.setup.n_class)
        
        loss = loss_f(logits, batch['label'], params)
        top1_acc, top2_acc = utils.topK_acc(logits, one_hot_labels, K = [1,2])
        total = batch['label'].shape[0]
        metrics = {'loss': loss*total, 'top1_acc' : top1_acc*100., 'top2_acc' : top2_acc*100., 'total': total }
        state, metrics = jmp.cast_to_full((state, metrics))
        return jax.lax.psum(metrics, axis_name='batch')
    
    @functools.partial(jax.pmap, axis_name='batch')
    def eval_output(run_state, batch):
        params, state, _, _, rng = run_state
        #rng, new_rng = jax.random.split(rng)
        logits, _ = forward.apply(params, state, rng, batch, training = False)
        return logits
    
    
    rng = jax.random.PRNGKey(config.setup.seed)
    rng = jnp.broadcast_to(rng, (local_device_count,) + rng.shape)
    
    if config.mode.restore():
        run_state = jax.pmap(restore_from_file)(rng)
    else:
        run_state = jax.pmap(initial_state)(rng, next(datasets['train']))
    
    
    model_info = make_model_info(run_state.params)
    #summary = hk.experimental.tabulate(train_step, columns = ['module', 'params_size', 'params_bytes'], tabulate_kwargs={'tablefmt':'pipe'})(train_state, batch)
    summary = None
    if jax.process_index() == 0:
        config.print_summary(summary, data_info, model_info, config.phases)
    
    
    if config.mode.training():
        logger.info(f'starting training')
        for epoch in range(1, config.setup.n_epochs+1):
            with tqdm(**artifact.pbar_args) as pbar:
                pbar.set_description(artifact.pbar_desc(epoch))
                
                with artifact.timer('train', epoch):
                    for step in range(data_info['train']['n_step_per_epoch']):
                        data = next(datasets['train'])
                        run_state = update(run_state, data)
                        if step % config.info_step == 0:
                            metrics = eval_step(run_state, data)
                            metrics = jax.tree_map(lambda x: x[0], metrics)
                            artifact.set_metrics('train', metrics = metrics)
                            pbar.set_postfix_str(artifact.pbar_update())
                        pbar.update(1)
                
                for _ in range(data_info['test']['n_step_per_epoch']):
                    metrics = eval_step(run_state, next(datasets[phase]))
                    metrics = jax.tree_map(lambda x: x[0], metrics)
                    artifact.set_metrics('test', metrics = metrics)
                    pbar.update(1)
                
                if artifact.end_of_epoch(epoch):
                    copy_state = deepcopy(run_state)
                    c_params, c_state, c_opt_state, c_loss_scale, c_rng = copy_state
                    
                    c_opt_state = jax.tree_map(lambda x: x[0], c_opt_state)
                    c_params = jax.tree_map(lambda x: x[0], c_params)
                    artifact.save_checkpoint(RunState(c_params, c_state, c_opt_state, c_loss_scale, c_rng))
                
                pbar.set_postfix_str(artifact.postfix)
                pbar.update(1)
    
    
    if config.mode.eval():
        #from loss import make_softmax_ce_loss
        #loss_f = make_softmax_ce_loss(config)
        
        logger.info(f'starting evaluating')
        for _ in range(data_info['test']['n_step_per_epoch']):
            data = next(datasets['test'])
            logit = eval_output(run_state, data)
            
            artifact.add_x(jax.device_get(data['prop']),  'prop')
            artifact.add_x(jax.device_get(data['label']), 'label', shape = [data['label'].shape[0]*data['label'].shape[-1]] )
            artifact.add_x(jax.device_get(logit),         'logit')
        
        artifact.save_outputs('eval-outputs')
    
    


if __name__ == "__main__":
    #from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script for running for model training.')
    
    parser.add_argument('-c',  '--config',       action = 'store', dest = 'config',     type = str, default = 'config/config.yaml')
    parser.add_argument('-s',  '--seed',         action = 'store', dest = 'seed',       type = int, default = None)
    parser.add_argument('-m',  '--model',        action = 'store', dest = 'model',      type = str, default = None)
    parser.add_argument('-r',  '--id',           action = 'store', dest = 'id',         type = str, default = 'test1')
    parser.add_argument('-md', '--mode',         action = 'store', dest = 'mode',       type = str, default = 'nominal')
    #parser.add_argument('-sm', '--saved_model',  action = 'store', dest = 'saved_model',type = str, default = None)
    parser.add_argument('-mn', '--model_name',   action = 'store', dest = 'model_name', type = str, default = 'nominal')
    #parser.add_argument('-tb', '--tb_log',       action = 'store', dest = 'tb_log',     type = str, default = None)
    
    opts = parser.parse_args()
    
    main(opts)
    
    #test_checkpoint()
    
    
    
    
    
    
    
    
    