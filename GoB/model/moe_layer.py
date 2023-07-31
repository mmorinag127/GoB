import haiku as hk
import jax
import jax.numpy as jnp
from .token_mixing import naive_top_k
import einops


def _make_moe_layer(layer, name):
    def moe_layer(x, training, gamma = None, beta = None):
        if hk.running_init:
            init_rng = hk.next_rng_key() #if hk.running_init() else None
            init_rng = jax.random.split(init_rng, x.shape[0])
        inner = hk.transform(layer, apply_rng=True)
        params = hk.lift(jax.vmap(inner.init, in_axes = (0, 0, None, None, None)), name=name)(init_rng, x, training, gamma = gamma, beta = beta)
        
        rng = hk.next_rng_key()
        rng = jax.random.split(rng, x.shape[0])
        
        outputs = jax.vmap(inner.apply, in_axes=(0, 0, 0, None))(params, rng, x, training)
        return outputs
    return moe_layer

def make_moe_layer(layer, name, n_experts):
    init_rng = None
    if hk.running_init():
        init_rng = hk.next_rng_key() #if hk.running_init() else None
        init_rng = jax.random.split(init_rng, n_experts)
    
    def wrapper(x, training):
        return layer()(x, training=training)
    
    inner = hk.without_apply_rng(hk.transform(wrapper))
    init_model = hk.lift(jax.vmap(inner.init, in_axes = (0, 0, None)))
    #init_model = hk.experimental.transparent_lift(jax.vmap(inner.init, in_axes = (0, 0, None)) )
    
    def moe_layer(x, *args, training=True, **kwargs):
        params = init_model(init_rng, x, training)
        outputs = jax.vmap(inner.apply, in_axes=(0, 0, None))(params, x, training)
        return outputs
    #return hk.to_module(moe_layer)(name=name)
    return moe_layer

def expert_label_importance_loss(n_class, gates_softmax, labels, mask=None):
    sg = jnp.sum(gates_softmax, axis=1, where = mask)
    one_hot = jax.nn.one_hot(labels, n_class)
    ce = jnp.einsum('BC,BE -> CE', one_hot, sg)
    importance = jnp.sum(ce, axis=0)
    imp_std = jnp.std(importance)
    imp_mean = jnp.mean(importance)
    imp_loss = (imp_std/imp_mean)**2
    return imp_loss

def importance_loss(gates_softmax, mask = None, eps = 1.0e-5):
    importance = jnp.sum(gates_softmax, axis = (0,1), where=mask)
    imp_std = jnp.std(importance)
    imp_mean = jnp.mean(importance)
    imp_loss = (imp_std/imp_mean)**2
    return imp_loss

def expert_load_loss(gates_softmax, T, topk_softmax, n_experts, mask = None):
    # from https://arxiv.org/pdf/2101.03961.pdf, eq4-6
    # T = jnp.sum(mask)
    f = jnp.sum(gates_softmax, where=mask, axis=tuple(range(gates_softmax.ndim - 1)))/T
    P = jnp.sum(topk_softmax, where=mask, axis=tuple(range(topk_softmax.ndim - 1)))/T
    return n_experts*jnp.sum(f*P)

def make_sparse_noisy_topk_gating(layer, name, n_experts = 4, topK = 1, capacity = 4, loss_w_imp = 0.5, loss_w_load = 0.5, **kwargs):
    noise_std = 1.0
    noise_std_scale = (1.0 / n_experts) * noise_std
    def sparse_noisy_topk_gating(inputs, *args, training=True, mask = None, **kwargs):
        _, n_tokens, Din  = inputs.shape
        # L2 norm
        W = hk.get_parameter(f'{name}_w', [Din, n_experts], init = lambda shape, dtype: jnp.ones(shape, dtype))
        
        #W_noise = hk.get_parameter(f'{name}_w_noise', [1], init = lambda shape, dtype: jnp.ones(shape, dtype))
        #W /= jnp.linalg.norm(W, ord=2, axis=-1, keepdims=True)
        
        #gates_inputs = inputs/jnp.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
        gates_inputs = inputs
        
        mask_D, mask_E = None, None
        if mask is not None:
            mask_E = einops.repeat(mask, 'b n () -> b n d', d = n_experts)
            mask_D = einops.repeat(mask, 'b n () -> b n d', d = inputs.shape[-1])
            gates_inputs = jnp.where(mask_D, inputs, 0.0)
        
        gates_logits = jnp.dot(gates_inputs, W)
        if mask_E is not None:
            gates_logits = jnp.where(mask_E, gates_logits, 0.)
        # gates_logits /= jnp.linalg.norm(gates_logits, ord=2, axis=-1, keepdims=True)
        
        load_loss = 0.
        
        if noise_std > 0.0 and training and loss_w_load > 0.0:
            
            rng = hk.next_rng_key() if hk.running_init() else None
            # noise_std_scale = abs(W_noise)
            gates_logits_noisy = gates_logits + noise_std_scale * jax.random.normal(rng, shape=gates_logits.shape)
            if mask_E is not None:
                gates_logits_noisy = jnp.where(mask_E, gates_logits_noisy, 0.)
            
            # load loss
            topk_index = naive_top_k(gates_logits_noisy, topK, axis = -1)
            threshold_index = topk_index[..., -1]
            topk_onehot = jax.nn.one_hot(threshold_index, n_experts)
            
            #threshold_item = jnp.sum(val*logits_noise, axis = -1)
            threshold_item = jnp.sum(topk_onehot*gates_logits_noisy, axis = -1)
            noise_required_to_win = (threshold_item[..., None] - gates_logits)/noise_std_scale
            p = 1. - jax.scipy.stats.norm.cdf(noise_required_to_win)
            p_mean = jnp.mean(p, axis = 1, where=mask_E)
            load_std = jnp.std(p_mean)
            load_mean = jnp.mean(p_mean)
            load_loss = (load_std/load_mean)**2
            gates_logits = gates_logits_noisy
        
        # tau = hk.get_parameter(f'{name}_softmax_tau', [1], init = lambda shape, dtype: jnp.ones(shape, dtype))
        # gates_logits = gates_logits/tau
        
        if mask_E is not None:
            dtype = gates_logits.dtype
            big_neg = jnp.finfo(dtype).min
            gates_logits = jnp.where(mask_E, gates_logits, big_neg)
        
        gates_softmax = jax.nn.softmax(gates_logits)
        if mask_E is not None:
            gates_softmax = jnp.where(mask_E, gates_softmax, 0.0)
        
        # pick up only topK element(expert index) for each token
        topk_index = naive_top_k(gates_softmax, k=topK, axis=-1)
        topk_onehot = jnp.sum(jax.nn.one_hot(topk_index, n_experts, axis = -1), axis = -2)
        
        gates_softmax = gates_softmax * topk_onehot
        if mask_E is not None:
            gates_softmax = jnp.where(mask_E, gates_softmax, 0.0)
        
        # importance loss
        imp_loss = 0.
        if loss_w_imp > 0.0:
            imp_loss = importance_loss(gates_softmax, mask)
        
        # total aux loss
        aux = {'aux_loss':loss_w_imp*imp_loss + loss_w_load*load_loss, 'imp_loss':imp_loss, 'load_loss':load_loss, 'moe_score': gates_softmax}
        
        topC_index  = naive_top_k(gates_softmax, k=capacity, axis = 1)
        dispatch_w  = jax.nn.one_hot(topC_index, n_tokens, axis=1, dtype=jnp.bool_)
        combine_w   = jnp.einsum("BSE,BSEC->BSEC", gates_softmax, dispatch_w, precision = jax.lax.Precision.DEFAULT)
        moe_inputs  = jnp.einsum("BSEK,BSD->EBKD", dispatch_w, inputs, precision = jax.lax.Precision.DEFAULT)
        moe_outputs = make_moe_layer(layer, name, n_experts)(moe_inputs, training)
        outputs     = jnp.einsum("BSEK,EBKD->BSD", combine_w, moe_outputs, precision = jax.lax.Precision.DEFAULT)
        
        return outputs, aux
    return sparse_noisy_topk_gating

def make_sparse_x_moe(layer, name, n_experts = 4, topK = 1, capacity = 4, loss_w_load = 0.5, dim_e_factor = 2, eps = 1.0e-5, **kwargs):
    De = n_experts//dim_e_factor
    def sparse_x_moe(inputs, *args, training=True, mask = None, **kwargs):
        _, n_tokens, Din  = inputs.shape
        # L2 norm
        
        # ones_init = lambda shape, dtype: jnp.ones(shape, dtype)
        W = hk.get_parameter(f'{name}_w', [Din, De], init = hk.initializers.Constant(1.0))
        E = hk.get_parameter(f'{name}_e', [De, n_experts], init = hk.initializers.RandomUniform(0.1, 1.0))
        tau = hk.get_parameter(f'{name}_tau', [1], init = hk.initializers.Constant(1.0))
        
        #gates_inputs = inputs/jnp.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
        mask_D, mask_E = None, None
        if mask is not None:
            mask_E = einops.repeat(mask, 'b n () -> b n d', d = n_experts)
            mask_D = einops.repeat(mask, 'b n () -> b n d', d = inputs.shape[-1])
            gates_inputs = jnp.where(mask_D, inputs, 0.0)
        
        Wh = jnp.dot(gates_inputs, W)
        gates_logits = jnp.dot(Wh, E)
        # gates_logits /= (jnp.linalg.norm(Wh, ord=2,axis=-1, keepdims=True)+eps)
        gates_logits /= (jnp.linalg.norm(E, ord=2, axis= 0, keepdims=True)+eps)
        
        if mask_E is not None:
            gates_logits = jnp.where(mask_E, gates_logits, 0.)
        
        gates_softmax = jax.nn.softmax(gates_logits/tau, axis=-1)
        if mask_E is not None:
            gates_softmax  = jnp.where(mask_E, gates_softmax, 0.0)
        
        topk_index = naive_top_k(gates_softmax, k=topK, axis=-1)
        topk_onehot = jnp.squeeze(jax.nn.one_hot(topk_index, n_experts, axis = -1), axis = -2)
        
        # load loss
        # from https://arxiv.org/pdf/2101.03961.pdf, eq4-6
        load_loss = 0.
        if loss_w_load > 0.0:
            T = jnp.sum(mask)
            f = jnp.sum(gates_softmax, where=mask_E, axis=(0,1))/T
            P = jnp.sum(topk_onehot, where=mask_E, axis=(0,1))/T
            load_loss = n_experts*jnp.sum(f*P)
        
        gates_score = gates_softmax * topk_onehot
        
        if mask_E is not None:
            gates_score = jnp.where(mask_E, gates_score, 0.0)
        
        # total aux loss
        aux = {'aux_loss':loss_w_load*load_loss, 'imp_loss':0, 'load_loss':load_loss, 'moe_score': gates_softmax}
        
        topC_index  = naive_top_k(gates_score, k=capacity, axis = 1)
        dispatch_w  = jax.nn.one_hot(topC_index, n_tokens, axis=1, dtype=jnp.bool_)
        combine_w   = jnp.einsum("BSE,BSEC->BSEC", gates_score, dispatch_w, precision = jax.lax.Precision.DEFAULT)
        moe_inputs  = jnp.einsum("BSEK,BSD->EBKD", dispatch_w, inputs, precision = jax.lax.Precision.DEFAULT)
        moe_outputs = make_moe_layer(layer, name, n_experts)(moe_inputs, training)
        outputs     = jnp.einsum("BSEK,EBKD->BSD", combine_w, moe_outputs, precision = jax.lax.Precision.DEFAULT)
        
        return outputs, aux
    return sparse_x_moe

def make_expert_choice_moe(layer, name, n_experts = 4, topK = 1, loss_w_imp=0.0, loss_w_load=0.0, loss_w_label = 0.0, **kwargs):
        
    def expert_choice_moe(inputs, *args, training = True, mask = None, **kwargs):
        _, n_tokens, Din  = inputs.shape
        # ones_init = lambda shape, dtype: jnp.ones(shape, dtype)
        W = hk.get_parameter(f'w', [Din, n_experts], init = hk.initializers.TruncatedNormal())
        
        mask_D, mask_E = None, None
        if mask is not None:
            mask_E = einops.repeat(mask, 'b n () -> b n d', d = n_experts)
            mask_D = einops.repeat(mask, 'b n () -> b n d', d = inputs.shape[-1])
            inputs = jnp.where(mask_D, inputs, 0.0)
        
        gates_logits = jnp.dot(inputs, W)
        if mask_E is not None:
            gates_logits = jnp.where(mask_E, gates_logits, 0.)
        
        
        gates_softmax = jax.nn.softmax(gates_logits, axis=-1)
        
        if mask_E is not None:
            gates_softmax  = jnp.where(mask_E, gates_softmax, 0.0)
        
        # load loss
        G, I = jax.lax.top_k(jnp.transpose(gates_softmax, axes=(0,2,1)), k=topK)
        P = jax.nn.one_hot(I, num_classes = n_tokens, axis=-1, dtype=jnp.int64)
        
        # importance loss
        imp_loss = importance_loss(gates_softmax, mask)
        topk_softmax = jnp.sum(jnp.transpose(P, axes=(0,3,2,1)), axis=-2)
        load_loss = expert_load_loss(gates_softmax, jnp.sum(mask), topk_softmax, n_experts, mask=mask_E)
        #label_balance_loss = expert_label_balance_loss(gates_softmax, labels, mask)
        
        # total aux loss
        aux = {'moe_score': gates_softmax, 'imp_loss':imp_loss, 'load_loss':load_loss}#, 'label_loss':label_balance_loss}
        if loss_w_imp > 0.0 or loss_w_load > 0.0 or loss_w_label > 0.0:
            aux['aux_loss'] = loss_w_imp*imp_loss+loss_w_load*load_loss# + loss_w_label*label_balance_loss
        
        moe_inputs  = jnp.einsum("BSD,BEKS -> EBKD", inputs, P)
        moe_outputs = make_moe_layer(layer, name, n_experts)(moe_inputs, training=training)
        outputs     = jnp.einsum("BEKS,BEK,EBKD -> BSD", P,G,moe_outputs)
        return outputs, aux
    return hk.to_module(expert_choice_moe)(name=name)

def make_moe_block(kwargs):
    if kwargs is None:
        return lambda layer, idx, name: layer
    return make_moe_block_(**kwargs)

def make_moe_block_(moe = None, cycle = None, **kwargs):
    if moe is None:
        return lambda layer, idx, name: layer
    
    if moe == 'SMoE':
        moe_layer = make_sparse_noisy_topk_gating
    elif moe == 'XMoE':
        moe_layer =  make_sparse_x_moe
    elif moe == 'ECMoE':
        moe_layer =  make_expert_choice_moe
    else:
        raise ValueError(f'moe_type:{moe} is not supported!!')
    
    if cycle is None or type(cycle) is not int:
        raise ValueError(f'cycle: should be int nor None')
    
    def make_layer(layer, idx, name):
        if (idx+1) % cycle == 0:
            return lambda : moe_layer(layer, name, **kwargs)
        return layer
    return make_layer














