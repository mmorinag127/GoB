import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import math

def make_w_init(name, **kwargs):
    
    """
    glorot_uniform  VarianceScaling(1.0, "fan_avg", "uniform")
    glorot_normal   VarianceScaling(1.0, "fan_avg", "truncated_normal")
    lecun_uniform   VarianceScaling(1.0, "fan_in",  "uniform")
    lecun_normal    VarianceScaling(1.0, "fan_in",  "truncated_normal")
    he_uniform      VarianceScaling(2.0, "fan_in",  "uniform")
    he_normal       VarianceScaling(2.0, "fan_in",  "truncated_normal")
    """
    if name is None:
        return None
    
    if   name == 'glorot_uniform':
        return hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    elif name == 'glorot_normal':
        return hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
    elif name == 'xavier_uniform':
        return hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    elif name == 'xavier_normal':
        return hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
    elif name == 'lecun_uniform':
        return hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform')
    elif name == 'lecun_normal':
        return hk.initializers.VarianceScaling(1.0, 'fan_in', 'truncated_normal')
    elif name == 'he_uniform':
        return hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
    elif name == 'he_normal':
        return hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')
    elif name == 'ZerO':
        return make_ZerO_init()
    elif hasattr(hk.initializers, name):
        return getattr(hk.initializers, name)(**kwargs)
    raise ValueError(f'{name} is not supported in weight_init!!')

def _compute_fans(shape, fan_in_axes=None):
    """Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        if fan_in_axes is not None:
            # Compute fan-in using user-specified fan-in axes.
            fan_in = np.prod([shape[i] for i in fan_in_axes])
            fan_out = np.prod([s for i, s in enumerate(shape) if i not in fan_in_axes])
        else:
            # If no axes specified, assume convolution kernels (2D, 3D, or more.)
            # kernel_shape: (..., input_depth, depth)
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out

def make_xavier_init(gain, ftype):
    def f_init(shape, dtype):
        fan_in, fan_out = _compute_fans(shape)
        std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
        if ftype == 'normal':
            return std*jax.random.normal(hk.next_rng_key(), shape, dtype)
        elif ftype == 'uniform':
            a = np.sqrt(3.0) * std
            return jax.random.uniform(hk.next_rng_key(), shape, dtype, -a,a)
        raise ValueError(f'{ftype} is not supported in xavier_init...')
    return f_init

from scipy.linalg import hadamard
def make_ZerO_init():
    # Algorithm 1 in the paper.
    def f_init(shape, dtype):
        m = shape[0]
        n = shape[1]
        
        if m <= n:
            eye = jnp.eye(m, n, dtype=dtype)
            if eye.shape != shape:
                eye = jnp.broadcast_to(eye, shape)
            return eye
        elif m > n:
            clog_m = math.ceil(math.log2(m))
            p = 2**(clog_m)
            e1 = jnp.eye(m,p)
            h = hadamard(p)/(2**(clog_m/2))
            e2 = jnp.eye(p,n)
            return e1 @ h @ e2
    
    return f_init
