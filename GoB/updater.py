import jax
import jax.numpy as jnp
import haiku as hk
import optax
import jmp

from typing import NamedTuple

import utils
import numpy as np

class Metrics(NamedTuple):
    loss: np.ndarray
    g_norm: np.ndarray
    p_norm: np.ndarray

class RunState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    loss_scale: jmp.LossScale
    #metrics: Metrics
    # rng: jnp.DeviceArray
class Updater:
    def __init__(self, network, loss_f, optimizer, lr_scheduler, mp_policy, mp_nonfinite, init_loss_scale, workdir, n_class, flooding, aux_to_metric = None):
        self._network = network
        self._loss_f = loss_f
        self._optimizer = optimizer(lr_scheduler)
        self._mp_policy = mp_policy
        self._mp_nonfinite = mp_nonfinite
        self._init_loss_scale = init_loss_scale
        self._workdir = workdir
        self._n_class = n_class
        self.flooding = flooding
        self.aux_to_metric = aux_to_metric
        
    def restore_from_file(self, filename = 'checkpoint_best.pkl'):
        import pickle
        file = f'{self._workdir}/{filename}'
        with open(file, 'rb') as f:
            run_state = pickle.load(f)
        params, state, opt_state, loss_scale = run_state
        return RunState(params, state, opt_state, loss_scale)
    
    def initial_state(self, batch, rng, training):
        batch  = self._mp_policy.cast_to_compute(batch)
        params, state = self._network.init(rng, batch, training = training)
        # for k, v in utils.param_flatten(params).items():
        #     print(f'{k} : {v.shape} {v.dtype}')
        opt_state = self._optimizer.init(params)
        loss_scale = self._init_loss_scale()
        return RunState(params, state, opt_state, loss_scale)
    
    def eval_output(self, run_state, batch):
        params, state, _, _ = run_state
        logits, aux, state = self.apply(params, state, batch, training = False)
        loss = self.loss_f(logits, batch, params, aux)
        return logits, loss, aux
    
    def eval_metrics(self, run_state, batch):
        logits, loss, aux = self.eval_output(run_state, batch)
        one_hot_labels = jax.nn.one_hot(batch['label'], self._n_class)
        top1_acc, top2_acc = utils.topK_acc(logits, one_hot_labels, K = [1,2], is_divide=True)
        # total = batch['label'].shape[0]
        # metrics = {'loss': loss*total, 't1_acc' : top1_acc*100., 't2_acc' : top2_acc*100., 'total': total}
        metrics = {'loss': loss, 't1_acc' : top1_acc*100., 't2_acc' : top2_acc*100.}
        return metrics
    
    def apply(self, params, state, batch, training):
        params = self._mp_policy.cast_to_compute(params)
        batch  = self._mp_policy.cast_to_compute(batch)
        (logits, aux), state = self._network.apply(params, state, None, batch, training = training)
        return logits, aux, state
    
    def loss_f(self, logits, batch, params, aux):
        aux_loss = sum(aux['aux_loss']) if 'aux_loss' in aux.keys() else None
        loss = self._loss_f(logits, batch['label'], params, aux_loss = aux_loss)
        if self.flooding is not None:
            loss = jnp.abs(loss - self.flooding) + self.flooding
        return loss
    
    def loss_step(self, params, state, loss_scale, batch):
        logits, aux, state = self.apply(params, state, batch, training = True)
        loss = self.loss_f(logits, batch, params, aux)
        return loss_scale.scale(loss), (state, aux)
    
    def update_nominal(self, run_state, batch, with_grads = False):
        import alpa
        params, state, opt_state, loss_scale = run_state
        grad_fn = alpa.value_and_grad(self.loss_step, has_aux = True)
        (loss, (new_state, aux)), grads = grad_fn(params, state, loss_scale, batch)
        grads = self._mp_policy.cast_to_compute(grads)
        grads = loss_scale.unscale(grads)
        loss = loss_scale.unscale(loss)
        grads = self._mp_policy.cast_to_param(grads)
        
        updates, new_opt_state = self._optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        grads_finite = jmp.all_finite(grads)
        loss_scale = loss_scale.adjust(grads_finite)
        
        new_params, new_state, new_opt_state = jmp.select_tree(grads_finite, (new_params, new_state, new_opt_state), (params, state, opt_state))

        new_state  = jmp.cast_to_full(new_state)
        grad_norm  = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grads)]))
        param_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(new_params) if e.ndim > 1]))
        metrics = {'loss':loss, 'g_norm':grad_norm, 'p_norm':param_norm}
        
        for key in self.aux_to_metric:
            if key in aux.keys():
                metrics[key] = np.mean(sum(aux[key]))
        
        if with_grads:
            return RunState(new_params, new_state, new_opt_state, loss_scale), metrics, grads
        return RunState(new_params, new_state, new_opt_state, loss_scale), metrics
    
    def loss_data2vec(self, params, state, batch, target, training): 
        params = self._mp_policy.cast_to_compute(params)
        batch  = self._mp_policy.cast_to_compute(batch)
        (logits, aux), _ = self._network.apply(params, state, None, batch, training = training)
        loss = self.loss_f(logits, target)
        loss = jnp.sum(loss, axis=-1)
        
    def update_data2vec(self, run_state, batch):
        params, state, opt_state, loss_scale = run_state
        import alpa
        
        grad_fn = alpa.value_and_grad(self.loss_data2vec, has_aux=True)
        target = self.apply(ema_params, ema_state, batch, training=False,check=None)
        
    # def grow_neuron(self):
    
    def make_gsam_loss_step(self, loss_fn, lr_max, lr_min, rho_min, rho_max, alpha, adaptive_perturbation, minimize_fp, eps=1.0e-6):
        
        def gsam_grad(g1, g2, sign):
            g1_norm = jnp.sqrt(sum(jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(g1)))
            g1_normalized = jax.tree_map(lambda x: x / g1_norm, g1)
            g1_normalized_flatten, _ = jax.tree_util.tree_flatten(g1_normalized)
            g2_flatten, _  = jax.tree_util.tree_flatten(g2)
            g2_projection_norm = sum(jnp.vdot(p, q) for (p,q) in zip(g1_normalized_flatten, g2_flatten))
            g2_residual = jax.tree_map(lambda a, b: a - g2_projection_norm * b, g2, g1_normalized)
            # Get GSAM gradient.
            g_gsam = jax.tree_map(lambda a, b: a + sign * b * alpha, g1, g2_residual)
            return g_gsam
        
        def gsam_loss_step(self, run_state, batch, ):
            import alpa
            params, state, opt_state, loss_scale = run_state
            
            (l_clean, (_, _)), g_clean = alpa.value_and_grad(self.loss_step, hax_aux=True)(params, state, batch, training=True)
            g_clean_length = jnp.sqrt(sum(jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(g_clean)))
            
            if lr_max == lr_min:
                sam_rho = rho_max
            else:
                sam_rho = rho_min + (rho_max - rho_min) * (lr - lr_min) / (lr_max - lr_min)
            
            # Per-worker perturbation.
            if adaptive_perturbation:
                params_sam = jax.tree_map(lambda a, b: a + jnp.abs(a) * sam_rho * b / (g_clean_length + eps), params, g_clean)
            else:
                params_sam = jax.tree_map(lambda a, b: a +              sam_rho * b / (g_clean_length + eps), params, g_clean)
            
            # Get gradients at perturbed weights.
            (_, (_, _)), g_robust = alpa.value_and_grad(self.loss_step, hax_aux=True)(params_sam, state, batch, training=True)

            # Decompose g_clean onto parallel and vertical to g_robust or 
            # Decompose g_robust onto parallel and vertical to g_clean.
            g_gsam = gsam_grad(g_robust, g_clean, -1) if minimize_fp else gsam_grad(g_clean, g_robust, +1)
            # Always return the clean loss (rather than the perturbed loss).
            return l_clean, g_gsam
        
    
    
    
    
    










