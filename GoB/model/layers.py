import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import einops


def make_film_layer(dim, depth, norm = None, dropout = None, activation = None):
    def film_layer(x, training):
        out = x
        if norm is not None:
            out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(x)
        for _ in range(depth):
            out = hk.Linear(dim)(out)
            if dropout is not None:
                out = hk.dropout(hk.next_rng_key(), dropout if training else 0, out)
            if activation is not None:
                out = jax.nn.gelu(out)
        return out
    return film_layer



class Dropout(hk.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.keep_rate = 1.0 - rate
    def __call__(self, x, training = False):
        if training:
            return hk.dropout(hk.next_rng_key(), self.rate, x)
        return x / self.keep_rate

class DropPath(hk.Module):
    def __init__(self, rate, scale_by_keep = True, ):
        super().__init__()
        self.rate = rate
        self.scale_by_keep = scale_by_keep
        
    def __call__(self, x, training):
        return drop_path(hk.next_rng_key(), self.rate, x, training, self.scale_by_keep)

class LayerScale(hk.Module):
    def __init__(self, dim, init_val = None):
        super().__init__()
        # dim should be shape
        if init_val is not None:
            self.param = hk.get_parameter('layer_scale', dim, init = lambda shape, dtype: jnp.ones(shape, dtype)*init_val)
        else:
            self.param = 1
    
    def __call__(self, x):
        return self.param * x

def layer_scale(x, dim, init_val):
    if init_val is not None:
        param = hk.get_parameter('layer_scale', dim, init = lambda shape, dtype: jnp.ones(shape, dtype)*init_val)
        return param * x
    return x

