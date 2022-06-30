import jax
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
    rng: jax.random.PRNGKey


class Updater:
    def __init__(self, network, loss_f, optimizer, mp_policy, mp_nonfinite, init_loss_scale, workdir, n_class):
        self._network = network
        self._loss_f = loss_f
        self._optimizer = optimizer
        self._mp_policy = mp_policy
        self._mp_nonfinite = mp_nonfinite
        self._init_loss_scale = init_loss_scale
        self._workdir = workdir
        self._n_class = n_class
        
    def initial_state(self, rng, data):
        out_rng, init_rng =  jax.random.split(rng)
        params, state = self._network.init(init_rng, data, training = True)
        opt_state = self._optimizer.init(params)
        loss_scale = self._init_loss_scale()
        return RunState(params, state, opt_state, loss_scale, out_rng)

    def restore_from_file(self, rng):
        import pickle
        file = f'{self._workdir}/checkpoint_best.pkl'
        with open(file, 'rb') as f:
            run_state = pickle.load(f)
        params, state, opt_state, loss_scale, _ = run_state
        return RunState(params, state, opt_state, loss_scale, rng)

    def eval_step(self, run_state, batch):
        params, state, _, _, rng = run_state
        rng, new_rng = jax.random.split(rng)
        logits, _ = self._network.apply(params, state, new_rng, batch, training = False)
        one_hot_labels = jax.nn.one_hot(batch['label'], self._n_class)
        
        loss = self._loss_f(logits, batch['label'], params)
        top1_acc, top2_acc = utils.topK_acc(logits, one_hot_labels, K = [1,2])
        total = batch['label'].shape[0]
        metrics = {'loss': loss*total, 'top1_acc' : top1_acc*100., 'top2_acc' : top2_acc*100., 'total': total }
        state, metrics = jmp.cast_to_full((state, metrics))
        return jax.lax.psum(metrics, axis_name='batch')
    
    def eval_output(self, run_state, batch):
        params, state, _, _, rng = run_state
        logits, _ = self._network.apply(params, state, rng, batch, training = False)
        return logits
    
    def loss_step(self, params, state, rng, loss_scale, batch):
        logits, state = self._network.apply(params, state, rng, batch, training = True)
        loss = self._loss_f(logits, batch['label'], params)
        return loss_scale.scale(loss), (loss, state)
    
    def update_nominal(self, run_state, batch):
        params, state, opt_state, loss_scale, rng = run_state 
        
        rng, new_rng = jax.random.split(rng)
        grads, (_, new_state) = (jax.grad(self.loss_step, has_aux = True)(params, state, rng, loss_scale, batch))
        
        
        grads = self._mp_policy.cast_to_compute(grads)
        grads = loss_scale.unscale(grads)
        
        grads = jax.lax.pmean(grads, axis_name='batch')
        grads = self._mp_policy.cast_to_param(grads)
        
        updates, new_opt_state = self._optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        if self._mp_nonfinite:
            grads_finite = jmp.all_finite(grads)
            loss_scale = loss_scale.adjust(grads_finite)
            new_params, new_state, new_opt_state = jmp.select_tree(grads_finite, (new_params, new_state, new_opt_state), (params, state, opt_state))
        
        state = jmp.cast_to_full(state)
        return RunState(new_params, new_state, new_opt_state, loss_scale, new_rng)
    
    def update_sam(self, run_state, batch):
        params, state, opt_state, loss_scale, rng = run_state 
        
        rng, new_rng = jax.random.split(rng)
        grads, (_, new_state) = (jax.grad(self.loss_step, has_aux = True)(params, state, rng, loss_scale, batch))
        
        
        grads = self._mp_policy.cast_to_compute(grads)
        grads = loss_scale.unscale(grads)
        
        grads = jax.lax.pmean(grads, axis_name='batch')
        grads = self._mp_policy.cast_to_param(grads)
        
        updates, new_opt_state = self._optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        
        if self._mp_nonfinite:
            grads_finite = jmp.all_finite(grads)
            loss_scale = loss_scale.adjust(grads_finite)
            new_params, new_state, new_opt_state = jmp.select_tree(grads_finite, (new_params, new_state, new_opt_state), (params, state, opt_state))
        
        state = jmp.cast_to_full(state)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return RunState(new_params, new_state, new_opt_state, loss_scale, new_rng)
        
        








