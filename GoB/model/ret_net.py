import haiku as hk
import jax
from jax import numpy as jnp
import math

from .token_mixing import make_object_tokenizer, make_drop_path, make_layer_scale, make_drop_path_rate, make_ffn_layer
from .weight_init import make_w_init

def make_ret_net(model, depth, dim, expansion, n_heads, dropout, drop_path, layer_scale, msr_bias, ffn_bias, w_init, msr_winit, ffn_winit, **kwargs):
    dim_inner = dim*expansion
    
    
    drop_path_rate = make_drop_path_rate(depth, drop_path)
    
    def ret_net(feature, *args, past_kvs=None, mode='parallel', training=True, one_hot=None, mask=None, glob=None, offset=0, **kwargs):
        tokens, mask = make_object_tokenizer(dim, w_init, name='tokenizer')(feature, one_hot = one_hot, glob = glob, mask = mask)
        #rel_pos = make_ret_rel_pos(dim, n_heads, chunk_size)(tokens.shape[1], tokens.dtype, mode=mode)
        xpos = make_xpos(dim, n_heads)(tokens.shape[1], dtype=tokens.dtype, offset=offset, token_mask=mask, mode=mode)
        
        hidden_state = tokens
        if past_kvs is None:
            past_kvs = [None]*depth
        
        for idx in range(depth):
            msr_out, past_kvs[idx] = make_msr_layer(dim, expansion, dropout, msr_bias, n_heads, msr_winit)(hidden_state, xpos, past_kv=past_kvs[idx], training=training, mode=mode)
            hidden_state = make_residual(drop_path_rate=drop_path_rate[idx], layer_scale=layer_scale, name=f'msr_res')(hidden_state, msr_out, training=training)
            
            ffn_out = make_ffn_layer(dim, dim_inner, dropout, ffn_winit, name=f'ffn{idx}', with_aux=False)(hidden_state, training=training)
            hidden_state = make_residual(drop_path_rate=drop_path_rate[idx], layer_scale=layer_scale, name=f'ffn_res')(hidden_state, ffn_out, training=training)
        
        output = hk.LayerNorm(axis = -1, create_scale = True, create_offset = True, name='ln')(hidden_state)
        return output, past_kvs
    return ret_net

def make_residual(drop_path_rate=None, layer_scale=None, name='residual'):
    def residual_connection(x, residual, training=True):
        if drop_path_rate is not None and drop_path_rate > 0:
            residual = make_drop_path(drop_path_rate, name='droppath')(residual, training=training)
        if layer_scale is not None:
            residual = make_layer_scale(x.shape[-1], layer_scale, name='layerscale')(x)
        return x + residual
    return hk.to_module(residual_connection)(name=name)

def make_ret_rel_pos(dim, n_heads, chunk_size, name=f'rel_pos', **kwargs):
    angle = 1.0 / (10000**jnp.linspace(0,1, dim//n_heads//2))
    angle = jnp.reshape(jnp.tile(jnp.expand_dims(angle, axis=-1), [1,2]), [-1])
    decay = jnp.log(1 - 2**(-5 - jnp.arange(n_heads)) )
    def ret_rel_pos(slen, dtype, mode='parallel'):
        if mode == 'recurrent': 
            sin = jnp.sin(angle *(slen-1))
            cos = jnp.cos(angle *(slen-1))
            return (sin, cos), jnp.exp(decay)
        elif mode == 'chunkwise':
            index = jnp.arange(slen)
            sin = jnp.sin(index[:, None] * angle[None, :])
            cos = jnp.cos(index[:, None] * angle[None, :])
            
            big_pos = jnp.finfo(dtype).max
            block_index = jnp.arange(chunk_size)
            mask = jnp.tril(jnp.ones([chunk_size, chunk_size], dtype=jnp.bool_))
            mask = jnp.where(~mask, block_index[:,None] - block_index[None,:], big_pos)
            mask = jnp.exp(mask * decay[:, None, None])
            mask = jnp.nan_to_num(mask)
            scale = jnp.sqrt(jnp.sum(mask, axis=-1, keepdims=True))
            mask = mask / scale
            
            cross_decay = jnp.exp(decay * chunk_size)
            inner_decay = jnp.exp(decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[:, None, None]
            inner_decay = inner_decay[:, :, None] / (scale / scale[:, -1, None])
            return (sin, cos), (mask, cross_decay, inner_decay)
        elif mode == 'parallel':
            index = jnp.arange(slen)
            big_pos = jnp.finfo(dtype).max
            sin = jnp.sin(index[:, None] * angle[None, :])
            cos = jnp.cos(index[:, None] * angle[None, :])
            mask = jnp.tril(jnp.ones([slen, slen], dtype=jnp.bool_))
            mask = jnp.where(~mask, index[:, None] - index[None, :], big_pos)
            mask = jnp.exp(mask * decay[:, None, None])
            mask = jnp.nan_to_num(mask)
            mask = mask / jnp.sqrt(jnp.sum(mask, axis=-1, keepdims=True))
            return (sin, cos), mask
        raise ValueError(f'mode:{mode} is not supported in ret_rel_pos')
    return ret_rel_pos

def make_xpos(dim, n_heads, gamma=0.4):
    D = dim//n_heads
    theta = jnp.exp((-4/D) * jnp.arange(0, D, 2) * jnp.log(10))
    scale = (jnp.arange(0, D, 2)/D + gamma)/(1 + gamma)
    gamma = 1 - jnp.exp( (-5 - jnp.arange(0, n_heads))*jnp.log(2) )
    def xpos(seq_len, dtype, offset=0, token_mask=None, mode='parallel'):
        cos = jnp.repeat(jnp.cos(jnp.expand_dims(jnp.arange(0, seq_len) + offset, axis=1) * theta), 2, axis=-1)
        sin = jnp.repeat(jnp.sin(jnp.expand_dims(jnp.arange(0, seq_len) + offset, axis=1) * theta), 2, axis=-1)
        T   = jnp.repeat(scale**(jnp.expand_dims(jnp.arange(0, seq_len) + offset, axis=1)),         2, axis=-1)
        
        if mode == 'recurrent':
            decay = jnp.reshape(gamma, [-1,1,1])
            return sin, cos, T, (decay, )
        
        decay = jnp.log(1 - 2**(-5 - jnp.arange(n_heads, dtype=dtype)))
        index = jnp.arange(seq_len)
        big_pos = jnp.finfo(dtype).max
        mask = jnp.tril(jnp.ones([seq_len, seq_len], dtype=jnp.bool_))
        mask = jnp.where(mask, index[:, None] - index[None, :], big_pos)
        mask = jnp.exp(mask * decay[:, None, None])
        mask = jnp.nan_to_num(mask)
        if token_mask is not None:
            mask = mask * token_mask
        mask = mask / jnp.sqrt(jnp.sum(mask, axis=-1, keepdims=True))
        if mode == 'parallel':
            return sin, cos, T, mask
        
        if mode == 'chunkwise':
            inner_decay = jnp.expand_dims(gamma, axis=1)**(jnp.arange(seq_len)+1)
            seq_len_ = seq_len
            if token_mask is not None:
                inner_decay = inner_decay * jnp.swapaxes(token_mask, -2, -1)
                seq_len_ = jnp.sum(token_mask)
            chunk_decay = jnp.reshape(gamma,  [-1,1,1])**(seq_len_)
            A = jnp.reshape(jnp.flip(inner_decay/jnp.expand_dims(gamma, axis=1), axis=[1,]), [1,n_heads,seq_len,1,1]) # Not sure A is good name or not
            return sin, cos, T, (mask, inner_decay, chunk_decay, A)
        raise ValueError(f'mode: {mode} is not supported')
    return xpos

def make_msr_layer(dim, v_factor, dropout, with_bias, n_heads, w_init, name='msr_layer', **kwargs):
    w_init = make_w_init(w_init)
    dim_head = dim*v_factor // n_heads
    dim_key = dim // n_heads
    dscale = dim_key**-0.5
    
    def rotate(x):
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        return jnp.reshape(jnp.stack([-x2, x1], axis=-1), x.shape)
    
    def parallel_retention(qr, kr, v, decay_mask):
        #print(decay_mask)
        print(qr.shape)
        print(kr.shape)
        print(v.shape)
        print(decay_mask.shape)
        qk = qr @ jnp.swapaxes(kr,-1, -2) * dscale
        qk = qk * decay_mask
        qk = qk / jnp.clip(jnp.abs(jnp.sum(qk, axis=-1, keepdims=True)), a_max=1)
        #output = jnp.matmul(qk, v)
        output = qk @ v
        #output = jnp.swapaxes(output, 1,2)
        return output, None
    
    def recurrent_retention(qr, kr, v, past_kv, decay):
        if past_kv is None:
            past_kv = 0
        current_kv = decay*past_kv + jnp.swapaxes(kr, -1, -2) * v
        output = qr @ current_kv * dscale
        return output, current_kv
    
    def chunkwise_retention(qr, kr, v, past_kv, decay_mask, chunk_decay, inner_decay, A):
        _,_,seq_len,_ = qr.shape
        qk = qr @ jnp.swapaxes(kr, -1, -2) * dscale
        qk = qk * decay_mask
        qk = qk / jnp.clip(jnp.abs(jnp.sum(qk, axis=-1, keepdims=True)), a_max=1)
        
        inner_retention = qk @ v
        
        if past_kv is None:
            past_kv = 0
            cross_retention = 0
        else:
            cross_retention = (qr @ past_kv) * jnp.reshape(inner_decay, [1,n_heads,seq_len,1])*dscale
        
        output = inner_retention + cross_retention
        current_kv = chunk_decay * past_kv + jnp.sum(jnp.expand_dims(kr, axis=-1)*A*jnp.expand_dims(v, axis=-2), axis=2)
        return output, current_kv
    
    def msr_layer(x, xpos, past_kv=None, training=True, mode='parallel', **kwargs):
        B, seq_len,_ = x.shape
        sin, cos, T, decays = xpos
        
        q = hk.Linear(dim,          with_bias=with_bias, w_init=w_init, name=f'linear_q')(x)
        k = hk.Linear(dim,          with_bias=with_bias, w_init=w_init, name=f'linear_k')(x)
        v = hk.Linear(dim*v_factor, with_bias=with_bias, w_init=w_init, name=f'linear_v')(x)
        g = hk.Linear(dim*v_factor, with_bias=with_bias, w_init=w_init, name=f'linear_g')(x)
        q = jnp.swapaxes(jnp.reshape(q, [B,seq_len,n_heads,-1]), 1,2) # B,H,L,D
        k = jnp.swapaxes(jnp.reshape(k, [B,seq_len,n_heads,-1]), 1,2) # B,H,L,D
        
        qr = (q*cos + rotate(q)*sin)*T
        kr = (k*cos + rotate(k)*sin)/T
        
        v  = jnp.swapaxes(jnp.reshape(v,  [B,seq_len,n_heads,-1]), 1,2) # B,H,L,D
        
        if mode == 'parallel':
            output, current_kv = parallel_retention(qr, kr, v, decays)
        elif mode == 'recurrent':
            output, current_kv = recurrent_retention(qr, kr, v, past_kv, decays)
        elif mode == 'chunkwise':
            output, current_kv = chunkwise_retention(qr, kr, v, past_kv, *decays)
        else:
            raise ValueError(f'mode:{mode} is not supported in msr layer')
        output = jnp.reshape(jnp.swapaxes(output, 1,2), [B, seq_len, -1])
        output = jnp.reshape(output, [B*seq_len, -1])
        
        
        output = hk.GroupNorm(groups=n_heads, create_scale=True, create_offset=True, data_format='N...C')(output)
        output = jnp.reshape(output, [B,seq_len,dim_head*n_heads])
        output = jax.nn.silu(g) * output
        output = hk.Linear(dim, with_bias= with_bias, name=f'linear_out')(output)
        output = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0.0, output)
        
        return output, current_kv
    return hk.to_module(msr_layer)(name=name)

def make_ffn_layer(dim, dim_inner, with_bias, w_init=None, name='ffn', **kwargs):
    _w_init = make_w_init(w_init)
    def ffn_layer(x, *args, training=True, **kwargs):
        out = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False, name='ln')(x)
        out = hk.Linear(dim_inner, with_bias=with_bias, name='linear0', w_init=_w_init)(out)
        out = jax.nn.gelu(out)
        out = hk.Linear(dim, with_bias=with_bias, name='linear1', w_init=_w_init)(out)
        return out
    return hk.to_module(ffn_layer)(name=name)





