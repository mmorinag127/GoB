import jax
import jax.numpy as jnp
import haiku as hk
import optax
import jmp

from typing import NamedTuple

import utils

class RunState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    loss_scale: jmp.LossScale

class Updater:
    def __init__(self, network, loss_f, optimizer, mp_policy, mp_nonfinite, init_loss_scale, workdir, n_class, flooding):
        self._network = network
        self._loss_f = loss_f
        self._optimizer = optimizer
        self._mp_policy = mp_policy
        self._mp_nonfinite = mp_nonfinite
        self._init_loss_scale = init_loss_scale
        self._workdir = workdir
        self._n_class = n_class
        self.flooding = flooding
        
    def restore_from_file(self, filename = 'checkpoint_best.pkl'):
        import pickle
        file = f'{self._workdir}/{filename}'
        with open(file, 'rb') as f:
            run_state = pickle.load(f)
        params, state, opt_state, loss_scale = run_state
        return RunState(params, state, opt_state, loss_scale)
    
    def initial_state(self, batch, rng, training):
        params, state = self._network.init(rng, batch, training = training)
        opt_state = self._optimizer.init(params)
        loss_scale = self._init_loss_scale()
        return RunState(params, state, opt_state, loss_scale)
    
    def apply(self, params, state, rng, batch, training, check):
        out, state = self._network.apply(params, state, rng, batch, training = training, check = check)
        logits, aux = out
        return logits, aux, state
    
    def loss_f(self, logits, batch, params, aux):
        aux_loss = aux['aux_loss'] if 'aux_loss' in aux.keys() else None
        loss = self._loss_f(logits, batch['label'], params, aux_loss = aux_loss)
        if self.flooding is not None:
            loss = jnp.abs(loss - self.flooding) + self.flooding
        return loss
    
    def eval_output(self, run_state, batch, rng):
        params, state, _, _ = run_state
        logits, aux, state = self.apply(params, state, rng, batch, training = False, check = None)
        loss = self.loss_f(logits, batch, params, aux)
        return logits, loss, aux
    
    def eval_metrics(self, run_state, batch, rng):
        logits, loss, aux = self.eval_output(run_state, batch, rng)
        one_hot_labels = jax.nn.one_hot(batch['label'], self._n_class)
        top1_acc, top2_acc = utils.topK_acc(logits, one_hot_labels, K = [1,2])
        total = batch['label'].shape[0]
        metrics = {'loss': loss*total, 't1_acc' : top1_acc*100., 't2_acc' : top2_acc*100., 'total': total}
        return metrics
    
    def loss_step(self, params, state, loss_scale, batch, rng):
        logits, aux, state = self.apply(params, state, rng, batch, training = True, check = None)
        loss = self.loss_f(logits, batch, params, aux)
        return loss_scale.scale(loss), (state, aux)
    
    def update_nominal(self, run_state, batch, rng):
        import alpa
        params, state, opt_state, loss_scale = run_state
        grad_fn = alpa.grad(self.loss_step, has_aux = True)
        grads, (new_state, aux) = grad_fn(params, state, loss_scale, batch, rng)
        # grad_fn = alpa.value_and_grad(self.loss_step, has_aux = True)
        # aux, grads = grad_fn(params, state, loss_scale, batch, rng)
        
        grads = self._mp_policy.cast_to_compute(grads)
        grads = loss_scale.unscale(grads)
        updates, new_opt_state = self._optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        grads_finite = jmp.all_finite(grads)
        loss_scale = loss_scale.adjust(grads_finite)
        
        #zero_grads = jax.
        
        zero_grads = jax.tree_map(lambda x: jnp.zeros(shape = x.shape, dtype = x.dtype), grads)
        #print(zero_grads)
        new_params, new_state, new_opt_state, fgrads = jmp.select_tree(grads_finite, (new_params, new_state, new_opt_state, grads), (params, state, opt_state, zero_grads))
        new_state = jmp.cast_to_full(new_state)
        
        grad_norm  = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grads)]))
        gradf_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(fgrads)]))
        param_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(new_params) if e.ndim > 1]))
        
        
        
        
        metrics = {'g_norm':grad_norm, 'gf_norm':gradf_norm, 'p_norm':param_norm}
        if 'aux_loss' in aux.keys():
            metrics['a_loss'] = jnp.mean(sum(aux['aux_loss']))
        
        return RunState(new_params, new_state, new_opt_state, loss_scale), metrics, grads










