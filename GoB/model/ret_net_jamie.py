import haiku as hk
import jax
from jax import numpy as jnp
import math

from .token_mixing import make_object_tokenizer
from .weight_init import make_w_init

def make_ret_net(model, depth, dim, w_init):
    
    
    
    
    def ret_net(feature, *args, one_hot=None, mask=None, glob=None, **kwargs):
        tokens, mask = make_object_tokenizer(dim, w_init, name='tokenizer')(feature, one_hot = one_hot, glob = glob, mask = mask)
        
        
        
        for idx in range(depth):
            out = make_msr_layer()(out)
            out = make_ffn_layer()(out)
        return out
    return ret_net

def make_retention_layer(dim, w_init, gamma, name='retention'):
    w_init = make_w_init(w_init)
    def retention_layer(x, *args, training=True, n=None, s_n_1=None, **kwargs):
        _, N, _ = x.shape
        w_q = hk.get_parameter(name=f'W_Q', shape=[dim,dim], init=w_init)
        w_k = hk.get_parameter(name=f'W_K', shape=[dim,dim], init=w_init)
        w_v = hk.get_parameter(name=f'W_V', shape=[dim,dim], init=w_init)
        theta = hk.get_parameter(name=f'theta', shape=[dim], init=w_init)
        i = jnp.complex(0.0, 1.0)
        
        if training:
            D = jnp.zeros([N,N], dtype=x.dtype)
            for n in range(N):
                for m in range(N):
                    if n >= m:
                        D[n,m] = gamma**(n - m)
            if x.dtype != jnp.complex_type:
                pass
            
            ns = jnp.arange(1, N+1, dtype=x.dtype)
            ns = jnp.complex(ns, jnp.zeros_like(ns))
            
            Theta = jnp.stack([jnp.exp(i*n*theta) for n in ns], axis=0)
            Theta_bar = jnp.conj(Theta)
            
            Q = (x @ w_q) * jnp.expand_dims(Theta,     axis=0)
            K = (x @ w_k) * jnp.expand_dims(Theta_bar, axis=0)
            V = (x @ w_v)
            atn = (Q @ jnp.transpose(K, axes=[0,2,1])) * jnp.expand_dims(D, axis=0)
            return atn @ V
        
        Theta = jnp.exp(i*n*theta)
        Theta_bar = jnp.conj(Theta)
        Q = (x @ w_q) * Theta
        K = (x @ w_k) * Theta_bar
        V = (x @ w_v)
        s_n = gamma*s_n_1 * jnp.expand_dims(K, axis=2) @ jnp.expand_dims(V, axis=1)
        y = jnp.v(jnp.expand_dims(Q, axis=1) @ s_n, axis=1)
        
        return y, s_n
    return hk.to_module(retention_layer)(name=name)

def make_parallel_retention_layer(layer, n_heads, gamma, name):
    init_rng = None
    if hk.running_init():
        init_rng = hk.next_rng_key() #if hk.running_init() else None
        init_rng = jax.random.split(init_rng, n_heads)
    
    def wrapper(x, training, g):
        return layer(g)(x, training=training)
    
    inner = hk.without_apply_rng(hk.transform(wrapper))
    init_model = hk.lift(jax.vmap(inner.init, in_axes = (0, 0, None, 0)))
    #init_model = hk.experimental.transparent_lift(jax.vmap(inner.init, in_axes = (0, 0, None)) )
    
    def parallel_retention_layer(x, *args, training=True, **kwargs):
        params = init_model(init_rng, x, training, gamma)
        outputs = jax.vmap(inner.apply, in_axes=(0, 0, None))(params, x, training)
        return outputs
    return parallel_retention_layer

def make_group_norm(num_group, num_channel, eps=1e-5, name='group_norm'):
    def group_norm(x):
        

def make_msr_layer(dim, n_heads, w_init):
    w_init = make_w_init(w_init)
    head_dim = dim // n_heads
    
    ret_layer = lambda g: make_retention_layer(dim, w_init, g, )
    
    
    def msr_layer(x, *args, training=True, n=None, s_n_1=None, **kwargs):
        B, N, D = x.shape
        gammas = (1 - jnp.exp(jnp.linspace( math.log(1/32.), math.log(1/512.), dtype=x.dtype)) )
        w_g = hk.get_parameter(name=f'W_G', shape=[dim,dim], init=w_init)
        w_o = hk.get_parameter(name=f'W_O', shape=[dim,dim], init=w_init)
        
        if training:
            n_per_head = N // n_heads
            assert n_per_head * n_heads == N, f'N should be divisible by n_heads, please change n_heads'
            head_inputs = jnp.transpose(jnp.reshape(x, [B, N, head_dim, n_heads]), axes=[3,0,1,2])
            
            head_outputs = make_parallel_retention_layer(ret_layer, n_heads, gammas)(head_inputs)
            head_outputs = jnp.concatenate(head_outputs, axis=3)
            Y = make_group_norm()
            
            
            
        
        
        
        return
    return msr_layer


def make_ffn_layer():
    def ffn_layer():
        return
    return ffn_layer
