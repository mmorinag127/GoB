import haiku as hk
import jax
import jax.numpy as jnp
from .token_mixing import naive_top_k

def make_moe_layer(layer, name):
    def moe_layer(x, training, gamma = None, beta = None):
        init_rng = hk.next_rng_key() #if hk.running_init() else None
        init_rng = jax.random.split(init_rng, x.shape[0])
        inner = hk.transform(layer, apply_rng=True)
        params = hk.lift(jax.vmap(inner.init, in_axes = (0, 0, None, None, None)), name=name)(init_rng, x, training, gamma = gamma, beta = beta)
        
        rng = hk.next_rng_key()
        rng = jax.random.split(rng, x.shape[0])
        
        outputs = jax.vmap(inner.apply, in_axes=(0, 0, 0, None))(params, rng, x, training)
        return outputs
    return moe_layer

def make_MoE_layer_token(layer, name, n_experts = 4, topK = 1, capacity = 4, loss_weights = {'imp':0.5, 'load':0.5}, **kwargs):
    noise_std = 1.0
    def MoE_layer(inputs, training, gamma = None, beta = None, check = None):
        _, n_tokens, Din  = inputs.shape
        # L2 norm
        W = hk.get_parameter(f'{name}_L2norm_W', [Din, n_experts], init = lambda shape, dtype: jnp.ones(shape, dtype))
        
        W /= jnp.linalg.norm(W, ord=2, axis=-1, keepdims=True)
        #inputs /= jnp.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
        gates_inputs = inputs/jnp.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
        gates_logits = jnp.dot(gates_inputs, W)
        
        load_loss = 0.
        if noise_std > 0.0 and training and loss_weights['load'] > 0.0:
            noise_std_scale = (1.0 / n_experts) * noise_std
            gates_logits_noisy = gates_logits + noise_std_scale * jax.random.normal(key = hk.next_rng_key(), shape=gates_logits.shape)
            
            # load loss
            index = naive_top_k(gates_logits_noisy, topK, axis = -1)
            threshold_index = index[..., -1]
            val = jax.nn.one_hot(threshold_index, n_experts)
            
            #threshold_item = jnp.sum(val*logits_noise, axis = -1)
            threshold_item = jnp.sum(val*gates_logits_noisy, axis = -1)
            noise_required_to_win = threshold_item[..., None] - gates_logits
            noise_required_to_win /= noise_std_scale
            
            p = 1. - jax.scipy.stats.norm.cdf(noise_required_to_win)
            p_mean = jnp.mean(p, axis = 1)
            
            load_std = jnp.std(p_mean)
            load_mean = jnp.mean(p_mean)
            load_loss = (load_std/load_mean)**2
            #load_loss = jnp.mean(load_loss)
            gates_logits = gates_logits_noisy
        
        tau = hk.get_parameter(f'{name}_softmax_tau', [n_experts], init = lambda shape, dtype: jnp.ones(shape, dtype))
        gates_softmax = jax.nn.softmax(gates_logits/tau)
        
        # pick up only topK element(expert index) for each token
        topk_index = naive_top_k(gates_softmax, k=topK, axis=-1)
        topk_onehot = jnp.sum(jax.nn.one_hot(topk_index, n_experts, axis = -1), axis = -2)
        
        #gates_softmax = jnp.einsum("BSE,BSE->BSE", gates_softmax, topk_onehot, precision = jax.lax.Precision.DEFAULT)
        gates_softmax = gates_softmax * topk_onehot
        
        # importance loss
        imp_loss = 0.
        if loss_weights['imp'] > 0.0:
            importance = jnp.sum(gates_softmax, axis = tuple(range(gates_softmax.ndim - 1)))
            imp_std = jnp.std(importance)
            imp_mean = jnp.mean(importance)
            imp_loss = (imp_std/imp_mean)**2
            #imp_loss = jnp.mean(imp_loss)
        
        # total aux loss
        aux = {'aux_loss':loss_weights['imp']*imp_loss + loss_weights['load']*load_loss}
        
        topC_index = naive_top_k(gates_softmax, k=capacity, axis = 1)
        dispatch_weights = jax.nn.one_hot(topC_index, n_tokens, axis=1, dtype=jnp.bool_)
        combine_weights = jnp.einsum("BSE,BSEC->BSEC", gates_softmax, dispatch_weights, precision = jax.lax.Precision.DEFAULT)
        moe_inputs = jnp.einsum("BSEK,BSD->EBKD", dispatch_weights, inputs, precision = jax.lax.Precision.DEFAULT)
        moe_outputs = make_moe_layer(layer, name)(moe_inputs, training, gamma = gamma, beta = beta)
        outputs = jnp.einsum("BSEK,EBKD->BSD", combine_weights, moe_outputs, precision = jax.lax.Precision.DEFAULT)
        
        if check is not None:
            aux['weights'] = gates_softmax
        
        return outputs, aux
    return MoE_layer


def make_MoE_layer_image(layer, n_experts = 4, topK = 1, capacity = 4, loss_weights = {'imp':0.5, 'load':0.5}, **kwargs):
    noise_std = 1.0
    
    def MoE_layer(inputs, training, check = None):
        B, n_tokens, Din  = inputs.shape
        # L2 norm
        W = hk.get_parameter('L2norm_W', [Din, n_experts], init = lambda shape, dtype: jnp.ones(shape, dtype))
        
        W /= jnp.linalg.norm(W, ord=2, axis=-1, keepdims=True)
        #inputs /= jnp.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
        gates_inputs = inputs/jnp.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
        gates_logits = jnp.dot(gates_inputs, W)
        gates_logits = jnp.mean(gates_logits, axis=1)
        
        load_loss = 0.
        if noise_std > 0.0 and training and loss_weights['load'] > 0.0:
            noise_std_scale = (1.0 / n_experts) * noise_std
            gates_logits_noisy = gates_logits + noise_std_scale * jax.random.normal(key = hk.next_rng_key(), shape=gates_logits.shape)
            
            # load loss
            index = naive_top_k(gates_logits_noisy, topK, axis = -1)
            threshold_index = index[..., -1]
            val = jax.nn.one_hot(threshold_index, n_experts)
            
            #threshold_item = jnp.sum(val*logits_noise, axis = -1)
            threshold_item = jnp.sum(val*gates_logits_noisy, axis = -1)
            noise_required_to_win = threshold_item[..., None] - gates_logits
            noise_required_to_win /= noise_std_scale
            
            p = 1. - jax.scipy.stats.norm.cdf(noise_required_to_win)
            p_mean = jnp.mean(p, axis = 1)
            
            load_std = jnp.std(p_mean)
            load_mean = jnp.mean(p_mean)
            load_loss = (load_std/load_mean)**2
            #load_loss = jnp.mean(load_loss)
            gates_logits = gates_logits_noisy
        
        tau = hk.get_parameter('softmax_tau', [n_experts], init = lambda shape, dtype: jnp.ones(shape, dtype))
        
        gates_softmax = jax.nn.softmax(gates_logits/tau)
        
        # pick up only topK element(expert index) for each image
        topk_index = naive_top_k(gates_softmax, k=topK, axis=-1)
        topk_onehot = jnp.sum(jax.nn.one_hot(topk_index, n_experts, axis = -1), axis = -2)
        
        #gates_softmax = jnp.einsum("BSE,BSE->BSE", gates_softmax, topk_onehot, precision = jax.lax.Precision.DEFAULT)
        gates_softmax = gates_softmax * topk_onehot
        
        # importance loss
        imp_loss = 0.
        if loss_weights['imp'] > 0.0:
            importance = jnp.sum(gates_softmax, axis = tuple(range(gates_softmax.ndim - 1)))
            imp_std = jnp.std(importance)
            imp_mean = jnp.mean(importance)
            imp_loss = (imp_std/imp_mean)**2
            #imp_loss = jnp.mean(imp_loss)
        
        # total aux loss
        aux = {'aux_loss':loss_weights['imp']*imp_loss + loss_weights['load']*load_loss}
        #capacity = B//n_experts
        topC_index = naive_top_k(gates_softmax, k=B//n_experts, axis = 0)
        dispatch_weights = jax.nn.one_hot(topC_index, B, axis=0, dtype=jnp.bool_)
        combine_weights = jnp.einsum("BE,BEC->BEC", gates_softmax, dispatch_weights, precision = jax.lax.Precision.DEFAULT)
        moe_inputs = jnp.einsum("BEC,BSD->ECSD", dispatch_weights, inputs, precision = jax.lax.Precision.DEFAULT)
        moe_outputs = make_moe_layer(layer)(moe_inputs, training)
        
        outputs = jnp.einsum("BEC,ECSD->BSD", combine_weights, moe_outputs, precision = jax.lax.Precision.DEFAULT)
        if check is not None:
            aux['expert_weights'] = gates_softmax#gates_logits
        return outputs, aux
    return MoE_layer






















