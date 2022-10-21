
from multiprocessing.sharedctypes import Value
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["JAX_ENABLE_X64"] = "1"
# os.environ["RAY_SCHEDULER_EVENTS"]="0"


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
#jnp.set_printoptions(threshold=2)
#jax.config.update("jax_debug_nans", True)




import mlflow
from tqdm.auto import tqdm

import utils
from config import Config
from model.utils import make_model, make_model_info
from dataset import dataset
from artifact import Artifact
from updater import Updater, RunState


def alpa_method(opts, config, updater):
    import alpa
    if opts.local:
        assert opts.n_hosts == 1
        alpa.init(cluster='local')
        physical_mesh = alpa.LocalPhysicalDeviceMesh(jax.local_devices()[:opts.n_devices_per_host])
        alpa.device_mesh.set_global_physical_mesh(physical_mesh)
        
    else:
        import ray
        alpa.init(cluster="ray")
        physical_mesh = alpa.get_global_cluster().get_physical_mesh(list(range(opts.n_hosts)), opts.n_devices_per_host)
    
    
    # as_option.force_data_parallel = True
    
    if 'pipeshard' in opts.alpa_type:
        logical_mesh_shape = (opts.n_devices_per_host,1)
        logical_mesh = physical_mesh.get_logical_mesh(logical_mesh_shape)
        as_option = alpa.AutoShardingOption()
        
        method = alpa.PipeshardParallel(
            devices = logical_mesh,
            num_micro_batches = config.setup.n_micro_batch,
            #auto_sharding_option = as_option,
            default_auto_sharding_option = as_option,
            layer_option = alpa.AutoLayerOption(layer_num=4),
            stage_option = alpa.AutoStageOption(),
        )
    elif 'data' in opts.alpa_type:
        method = alpa.DataParallel(num_micro_batches = config.setup.n_micro_batch)
    elif 'shard' in opts.alpa_type:
        method = alpa.ShardParallel(num_micro_batches = config.setup.n_micro_batch)
    else:
        raise ValueError(f'alpa type: {opts.alpa_type} is not supported!')
    
    grad_update = alpa.parallelize(updater.update_nominal, method = method, donate_argnums = (0,))
    eval_metrics = alpa.parallelize(updater.eval_metrics, method = alpa.DataParallel())
    return grad_update, eval_metrics

def make_batch_placement_specs(f, run_state, batch, rng):
    executable_train = f.get_executable(run_state, batch, rng)
    return executable_train.get_input_placement_specs()[1]

def make_alpa_dataitr(bps, dataset, config):
    import alpa
    it = map(lambda xs: jax.tree_map(lambda x: x._numpy(), xs),  dataset.ds)
    if bps is not None:
        it = alpa.DataLoader(it, bps, prefetch_size = config.dataset.n_prefetch)
        # it = alpa.MeshDriverDataLoader(dataset.batch_size, dataset.length, it, bps, prefetch_size = config.dataset.n_prefetch)
    return iter(it)

def main(opts):
    # config
    config = Config(opts, jax.process_index(), **{'setup:model': opts.model, 'setup:seed': opts.seed, 'setup:model_name': opts.model_name, 'setup:head':opts.head, 'setup:MoE':opts.MoE, 'setup:FiLM': opts.FiLM})
    
    # random seed
    utils.set_seed(config.setup.seed, logger = logger)
    
    make_mp_policy, initial_loss_scale = utils.make_mixed_precision(config)
    mp_policy = make_mp_policy()
    rng = jax.random.PRNGKey(config.setup.seed)
    init_rng, rng = jax.random.split(rng)
    
    
    datasets = {}
    data_length = 0
    
    for phase in config.phases:
        datasets[phase] = dataset.make_dataset(phase = phase, **config.dataset, dtype = mp_policy.compute_dtype)
        #datasets[phase].n_step = 5
        data_length += datasets[phase].n_step
    
    artifact = Artifact(config, data_length = data_length)
    updater = Updater(
        network = make_model(config, mp_policy), 
        loss_f = utils.make_loss(config), 
        optimizer = utils.make_optimizer(config, datasets['train'].n_step)(), 
        mp_policy = mp_policy,
        mp_nonfinite = config.mp_nonfinite, 
        init_loss_scale = initial_loss_scale, 
        workdir = config.workdir, n_class = config.setup.n_class
        )
    
    batch = {'label': jnp.ones((config.batch_size,), dtype = jnp.int32), 'image': jnp.ones((config.batch_size, *config.input.image_shape), dtype = mp_policy.compute_dtype),
             'prop': jnp.ones((config.batch_size, *config.input.prop_shape), dtype = mp_policy.compute_dtype)}
    #run_state = updater.restore_from_file() if config.mode.restore() else jax.jit(updater.initial_state)(batch, init_rng, True)
    run_state = updater.restore_from_file() if config.mode.restore() else updater.initial_state(batch, init_rng, True)
    
    
    out_rng, rng = jax.random.split(rng)
    
    if config.mode.training():
        grad_update, eval_metrics = alpa_method(opts, config, updater)
        datasets['train'].iter = make_alpa_dataitr(make_batch_placement_specs(grad_update,  run_state, batch, out_rng), datasets['train'], config)
        datasets['test'].iter  = make_alpa_dataitr(make_batch_placement_specs(eval_metrics, run_state, batch, out_rng), datasets['test'],  config)
    else:
        eval_output = jax.jit(updater.eval_output_with if opts.isMoE else updater.eval_output)
        datasets['test'].iter  = make_alpa_dataitr(None, datasets['test'],  config)
    
    #physical_mesh.sync_workers()
    
    model_info = make_model_info(run_state.params)
    summary = None
    config.print_summary(config.process_idx, {'experiment':config.experiment_name, 'run':config.run_name}, summary, datasets, model_info, config.phases, artifact.hparams)
    
    if config.mode.training():
        logger.info(f'starting training')
        
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
                monitor = utils.Monitor(20.0)#sec
                monitor.start_record()
                
                with artifact.timer('train', epoch):
                    for step in range(datasets['train'].n_step):
                        batch = next(datasets['train'].iter)
                        out_rng, rng = jax.random.split(rng)
                        run_state, run_metrics = grad_update(run_state, batch, out_rng)
                        
                        if step % config.info_step == 0:
                            metrics = eval_metrics(run_state, batch, out_rng)
                            run_metrics = {k:jax.device_get(v).item() for k,v in run_metrics.items()}
                            metrics.update(run_metrics)
                            artifact.set_metrics('train', metrics = metrics)
                        pbar.set_postfix_str(artifact.pbar_update())
                        pbar.update(1)
                
                monitor.stop_record()
                artifact.set_monitor('train', monitor.get_gpu_usage(mean=True))
                monitor = utils.Monitor(1.0)
                monitor.start_record()
                
                with artifact.timer('test', epoch):
                    for _ in range(datasets['test'].n_step):
                        batch = next(datasets[phase].iter)
                        metrics = eval_metrics(run_state, batch, out_rng)
                        metrics.update(run_metrics)
                        artifact.set_metrics('test', metrics = metrics)
                        pbar.set_postfix_str(artifact.pbar_update())
                        pbar.update(1)
                
                monitor.stop_record()
                artifact.set_monitor('test', monitor.get_gpu_usage(mean=True))
                
                if artifact.end_of_epoch(epoch,  jax.tree_map(lambda x: x[0], run_state.params)):
                    c_params, c_state, c_opt_state, c_loss_scale = deepcopy(run_state)
                    checkpoint_name = artifact.save_checkpoint(RunState(c_params, c_state, c_opt_state, c_loss_scale), name = 'checkpoint_best.pkl')
                    mlflow.log_artifact(checkpoint_name)
                
                pbar.set_postfix_str(artifact.postfix)
                pbar.update(1)
                if config.process_idx == 0:
                    mlflow.log_metrics(artifact.get_metrics(), step = epoch)
        
        c_params, c_state, c_opt_state, c_loss_scale = deepcopy(run_state)
        checkpoint_name = artifact.save_checkpoint(RunState(c_params, c_state, c_opt_state, c_loss_scale), name = 'checkpoint_last.pkl')
        mlflow.log_artifact(checkpoint_name)
        mlflow.end_run()
    #ray.stop()
    
    if config.mode.eval():
        logger.info(f'starting evaluating')
        with tqdm(total = datasets['test'].n_step, unit = 'batch', ncols = 250, bar_format="{desc}{percentage:3.0f}% {n_fmt: >5}/{total_fmt: >5} [{rate_fmt: >16}{postfix}]") as pbar:
            pbar.set_description()
            for _ in range(datasets['test'].n_step):
                data = next(datasets['test'].iter)
                if opts.isMoE:
                    logits, _, expert_weights = eval_output(run_state, data, rng)
                    expert_weights = jnp.swapaxes(expert_weights, 0, 1)
                    artifact.add_x(jax.device_get(expert_weights), 'expert_weights', shape = expert_weights.shape)
                else:
                    logits, _ = eval_output(run_state, data, rng)
                
                # print(expert_weights[0])
                # input('enter')
                labels = jax.device_get(data['label'])
                artifact.add_x(jax.device_get(data['prop']),  'prop')
                artifact.add_x(labels, 'label', shape = [labels.shape[0], ])
                artifact.add_x(jax.device_get(logits), 'logit')
                
                pbar.update(1)
        
        artifact.save_outputs('eval-outputs', xlists = ['label', 'logit', 'prop'])
    
    
    
    


if __name__ == "__main__":
    #from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script for running for model training.')
    
    parser.add_argument('-c',  '--config',       action = 'store', dest = 'config',             type = str,  default = 'config/config.yaml')
    parser.add_argument('-s',  '--seed',         action = 'store', dest = 'seed',               type = int,  default = None)
    parser.add_argument('-m',  '--model',        action = 'store', dest = 'model',              type = str,  default = None)
    parser.add_argument('-md', '--mode',         action = 'store', dest = 'mode',               type = str,  default = 'nominal')
    parser.add_argument('-mn', '--model_name',   action = 'store', dest = 'model_name',         type = str,  default = 'nominal')
    parser.add_argument('-he', '--head'      ,   action = 'store', dest = 'head',               type = str,  default = 'MLP-S-1')
    parser.add_argument('-mo', '--moe',          action = 'store', dest = 'MoE',                type = str,  default = None)
    parser.add_argument('-fl', '--film',         action = 'store', dest = 'FiLM',               type = str,  default = None)
    parser.add_argument('-e',  '--experiment',   action = 'store', dest = 'experiment_name',    type = str,  default = None)
    parser.add_argument('-r',  '--run_name',     action = 'store', dest = 'run_name',           type = str,  default = None)
    parser.add_argument('-l',  '--local',        action = 'store', dest = 'local',              type = bool, default = True)
    parser.add_argument('-nd', '--n_device',     action = 'store', dest = 'n_devices_per_host', type = int,  default = 1)
    parser.add_argument('-nh', '--n_hosts',      action = 'store', dest = 'n_hosts',            type = int,  default = 1)
    parser.add_argument('-gpu','--gpuid',        action = 'store', dest = 'gpuid',              type = int,  default = None)
    parser.add_argument('-moe','--isMoE',        action = 'store_true', dest = 'isMoE')
    parser.add_argument('-at', '--alpa_type',    action = 'store', dest = 'alpa_type',          type = str,  default = 'data')
    
    
    opts = parser.parse_args()
    
    main(opts)

    #test_checkpoint()
    
    
    
    
    
    
    
    
    