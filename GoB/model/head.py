import haiku as hk
import jax
from jax import numpy as jnp
import einops

from .token_mixing import make_attention_layer
from .weight_init import make_w_init




def make_head(model, **kwargs):
    
    if model == 'MLP':
        return make_mlp_ln_head(**kwargs)
    elif model == 'PN':
        return make_pn_head(**kwargs)
    raise ValueError(f'{type} is not supported !!')

def make_mlp_ln_head(n_classes, focal, w_init, name='mlp_head', **kwargs):
    def head(x, *args, training = True, mask=None, **kwargs):
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = True, name='ln')(x)
        
        b_init = hk.initializers.Constant(-2.0) if focal else None
        out = hk.Linear(output_size = n_classes, w_init = make_w_init(w_init), b_init=b_init, name=f'linear')(out)
        return out
    #return hk.to_module(head)(name=name)
    return head

def make_pn_head(dim, depth, dropout, n_classes, w_init, name='pn_head',**kwargs):
    def head(x, *args, training = True, mask=None, **kwargs):
        # x = hk.BatchNorm(True, True, 0.99, name=f'bn_pn_head')(x, is_training=training)
        
        for _ in range(depth):
            x = hk.Linear(dim, w_init=make_w_init(w_init))(x)
            x = jax.nn.relu(x)
            x = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0.0, x)
        x = hk.Linear(n_classes, w_init=make_w_init(w_init))(x)
        return x
    return head

def make_pooling(model, **kwargs):
    
    if model == 'GAP':
        return make_global_average_pooling(**kwargs)
    elif model == 'AP':
        return make_attention_pooling(**kwargs)
    elif model == 'GP':
        return make_global_token_pooling(**kwargs)
    raise ValueError(f'{model} is not supported !!')
    

def make_global_average_pooling(axis=1, name='gap', ):
    def pooling(x, *args, training=True, mask=None, **kwargs):
        out = jnp.mean(x, axis=axis, where=mask)
        return out
    #return hk.to_module(pooling)(name=name)
    return pooling

def make_attention_pooling(dim, expansion, dropout, n_heads, qkv_bias, w_init=None, k = 1, name='attention_pooling', **kwargs):
    dim_inner = dim * expansion
    def attention_pooling(x, *args, training = False, mask=None, **kwargs):
        B, _, _ = x.shape
        s = hk.get_parameter(name=f's', shape = [1, k, dim], init=hk.initializers.RandomNormal())
        s = einops.repeat(s, '() k d -> b k d', b=B)
        
        out = make_attention_layer(dim, dim_inner, dropout, n_heads, qkv_bias, w_init)(s, x, training, mask=mask)
        return out
    #return hk.to_module(attention_pooling)(name=name)
    return attention_pooling

def make_global_token_pooling(gidx=0):
    def global_token_pooling(x, *args, training=False, mask=None, **kwargs):
        out = x[:, gidx, :]
        return out
    return global_token_pooling