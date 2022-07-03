
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from logger import make_logger
logger = make_logger(__name__)

import tensorflow as tf
#tf.config.experimental.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:# Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(f'TF: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
    except RuntimeError as e: # Memory growth must be set before GPUs have been initialized
        print(e)
tf.get_logger().setLevel('ERROR')


import argparse
from copy import deepcopy

import jax
import jax.numpy as jnp
jnp.set_printoptions(threshold=2)
jax.config.update("jax_debug_nans", True)

import mlflow
from tqdm.auto import tqdm

import utils
from config import Config
from model.utils import make_model, make_model_info
from dataset import dataset
from artifact import Artifact
from updater import Updater, RunState


def main(opts):
    # config
    
    config = Config(opts, jax.process_index(), **{'setup:model': opts.model, 'setup:seed': opts.seed, 'setup:model_name': opts.model_name})
    local_device_count = jax.local_device_count()
    
    # random seed
    rng = jax.random.PRNGKey(config.setup.seed)
    rng = jnp.broadcast_to(rng, (local_device_count,) + rng.shape)
    utils.set_seed(config.setup.seed, logger = logger)
    
    make_mp_policy, initial_loss_scale = utils.make_mixed_precision(config)
    mp_policy = make_mp_policy()
    
    forward = make_model(config, mp_policy)
    
    datasets = {}
    data_length = 0
    
    for phase in config.phases:
        datasets[phase] = dataset.make_dataset(phase = phase, **config.dataset, dtype = mp_policy.compute_dtype)
        #datasets[phase].n_step = 20
        data_length += datasets[phase].n_step
    
    artifact = Artifact(config, data_length = data_length)
    
    optimizer = utils.make_optimizer(config, datasets['train'].n_step)()
    loss_f = utils.make_loss(config)
    
    updater = Updater(forward, loss_f, optimizer, mp_policy, config.mp_nonfinite, initial_loss_scale, config.workdir, config.setup.n_class)
    grad_update = jax.pmap(updater.update_nominal, axis_name='batch', donate_argnums=(0,))
    eval_step   = jax.pmap(updater.eval_step,      axis_name='batch')
    eval_output = jax.pmap(updater.eval_output,    axis_name='batch')
    
    if config.mode.restore():
        run_state = jax.pmap(updater.restore_from_file)(rng)
    else:
        run_state = jax.pmap(updater.initial_state)(rng, next(datasets['train'].iter))
    
    
    model_info = make_model_info(run_state.params)
    #summary = hk.experimental.tabulate(train_step, columns = ['module', 'params_size', 'params_bytes'], tabulate_kwargs={'tablefmt':'pipe'})(train_state, batch)
    summary = None
    if config.process_idx == 0:
        config.print_summary({'experiment':config.experiment_name, 'run':config.run_name}, summary, datasets, model_info, config.phases, artifact.hparams)
    
    if config.mode.training():
        logger.info(f'starting training')
        logger.info(f'monitor started')
        
        # mlflow 
        if config.process_idx == 0:
            mlflow.set_tracking_uri(f'{config.exp_dir}/mlruns')
            experiment = mlflow.set_experiment(experiment_name = config.experiment_name)
            mlflow.start_run(run_name = config.run_name)
            mlflow.log_param('seed', config.seed)
            mlflow.log_params(artifact.hparams)
            
        
        for epoch in range(1, config.setup.n_epochs+1):
            with tqdm(**artifact.pbar_args) as pbar:
                pbar.set_description(artifact.pbar_desc(epoch))
                monitor = utils.Monitor(0.1)#sec
                monitor.start_record()
                
                with artifact.timer('train', epoch):
                    for step in range(datasets['train'].n_step):
                        data = next(datasets['train'].iter)
                        run_state = grad_update(run_state, data)
                        if step % config.info_step == 0:
                            metrics = eval_step(run_state, data)
                            artifact.set_metrics('train', metrics = jax.tree_map(lambda x: x[0], metrics))
                            pbar.set_postfix_str(artifact.pbar_update())
                        pbar.update(1)
                
                monitor.stop_record()
                artifact.set_monitor('train', monitor.get_gpu_usage())
                
                monitor = utils.Monitor(1)#sec
                monitor.start_record()
                
                for _ in range(datasets['test'].n_step):
                    metrics = eval_step(run_state, next(datasets[phase].iter))
                    artifact.set_metrics('test', metrics = jax.tree_map(lambda x: x[0], metrics))
                    pbar.update(1)
                monitor.stop_record()
                artifact.set_monitor('test', monitor.get_gpu_usage())
                
                if artifact.end_of_epoch(epoch,  jax.tree_map(lambda x: x[0], run_state.params)):
                    c_params, c_state, c_opt_state, c_loss_scale, c_rng = deepcopy(run_state)
                    c_opt_state = jax.tree_map(lambda x: x[0], c_opt_state)
                    c_params = jax.tree_map(lambda x: x[0], c_params)
                    checkpoint_name = artifact.save_checkpoint(RunState(c_params, c_state, c_opt_state, c_loss_scale, c_rng))
                    mlflow.log_artifact(checkpoint_name)
                
                pbar.set_postfix_str(artifact.postfix)
                pbar.update(1)
                if config.process_idx == 0:
                    mlflow.log_metrics(artifact.get_metrics(), step = epoch)
                
        
        
        mlflow.end_run()
    artifact.close()
    
    if config.mode.eval():
        logger.info(f'starting evaluating')
        for _ in range(datasets['test'].n_step):
            data = next(datasets['test'].iter)
            logit = eval_output(run_state, data)
            
            artifact.add_x(jax.device_get(data['prop']),  'prop')
            artifact.add_x(jax.device_get(data['label']), 'label', shape = [data['label'].shape[0]*data['label'].shape[-1]] )
            artifact.add_x(jax.device_get(logit),         'logit')
        
        artifact.save_outputs('eval-outputs')
    
    
    


if __name__ == "__main__":
    #from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script for running for model training.')
    
    parser.add_argument('-c',  '--config',       action = 'store', dest = 'config',          type = str, default = 'config/config.yaml')
    parser.add_argument('-s',  '--seed',         action = 'store', dest = 'seed',            type = int, default = None)
    parser.add_argument('-m',  '--model',        action = 'store', dest = 'model',           type = str, default = None)
    parser.add_argument('-md', '--mode',         action = 'store', dest = 'mode',            type = str, default = 'nominal')
    parser.add_argument('-mn', '--model_name',   action = 'store', dest = 'model_name',      type = str, default = 'nominal')
    parser.add_argument('-e',  '--experiment',   action = 'store', dest = 'experiment_name', type = str, default = None)
    parser.add_argument('-r',  '--run',          action = 'store', dest = 'run_name',        type = str, default = None)
    
    
    opts = parser.parse_args()
    
    main(opts)
    
    #test_checkpoint()
    
    
    
    
    
    
    
    
    