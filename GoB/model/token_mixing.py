import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import einops

def naive_top_k(data, k, axis = -1):
    N = data.shape[axis]
    def top1(data_, unused):
        index = jnp.argmax(data_, axis=axis)
        mask = jax.nn.one_hot(index, N, axis = axis, dtype=jnp.bool_)
        data_wo_top1 = data_ * (1 - mask)
        return data_wo_top1, index
    index = []
    for i in range(k):
        data, idx = top1(data, None)
        index.append(idx)
        
    index = jnp.stack(index)
    index = jnp.moveaxis(index, 0, -1)
    return index


def make_patch_layer(patch_size, dim, name = 'patch_emb'):
    def patch_layer(image):
        #patch = einops.rearrange(image, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        B, H, W, C = image.shape
        h = H//patch_size
        w = W//patch_size
        P = patch_size
        patch = jnp.expand_dims(image, [2,3])
        patch = jnp.reshape(patch, [B,h,P,w,P,C])
        patch = jnp.swapaxes(patch, 2, 3)
        patch = jnp.reshape(patch, [B,h*w,P*P*C])
        patch = hk.Linear(dim, name = f'{name}_linear')(patch)
        patch = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True, name = f'{name}_layer_norm')(patch)
        return patch
    return patch_layer

def make_meta_layer(dim, mixing_layer, drop_path_rate, lscale, hax_aux = False, name = 'meta'):
    def meta_layer(x, training = True, gamma = None, beta = None):
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True, name=f'{name}_layer_norm')(x)
        out, aux = mixing_layer(out, training)
        if gamma is not None and beta is not None:
            gamma = hk.Linear(dim, name=f'{name}_film_linear0')(gamma)
            beta  = hk.Linear(dim, name=f'{name}_film_linear1')(beta)
            if x.ndim > 2:
                gamma = jnp.reshape(gamma, (gamma.shape[0], -1, gamma.shape[-1]))
                beta = jnp.reshape(beta, (beta.shape[0], -1, beta.shape[-1]))
            out = gamma * out + beta
        
        if lscale is not None:
            param = hk.get_parameter(name=f'{name}_layer_scale', shape = [dim], init = lambda shape, dtype: jnp.ones(shape, dtype)*lscale)
            out = param*out
        
        if drop_path_rate is not None:
            out = drop_path(hk.next_rng_key(), drop_path_rate if training else 0.0, out)
        
        return x + out, aux
    return meta_layer

def make_mixer_layer(dim, dim_inner, dropout_rate, n_heads, name = 'mixer',**kwargs):
    def mixer_layer(x, training):
        B, N, _ = x.shape
        dim_head = dim_inner // n_heads
        out = jnp.swapaxes(x, 1, 2)
        out = hk.Linear(dim_inner, name=f'{name}_linear0')(out)
        out = jnp.reshape(out, [B, dim, n_heads, dim_head])
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0.0, out)
        out = hk.Linear(N//n_heads, name=f'{name}_linear1')(out)
        out = jnp.reshape(out, [B, dim, N] )
        out = jnp.swapaxes(out, 1, 2)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0.0, out)
        return out, {}
    return mixer_layer

def make_attention_layer(dim, dim_inner, dropout_rate, n_heads, qkv_bias, name = 'self_attention', **kwargs):
    def attention_layer(x, training):
        B, N, _ = x.shape
        dim_head = dim_inner//n_heads
        q = jnp.reshape(hk.Linear(output_size = dim_inner, with_bias = qkv_bias, name=f'{name}_linear0')(x), [B, N, n_heads, dim_head])
        k = jnp.reshape(hk.Linear(output_size = dim_inner, with_bias = qkv_bias, name=f'{name}_linear1')(x), [B, N, n_heads, dim_head])
        v = jnp.reshape(hk.Linear(output_size = dim_inner, with_bias = qkv_bias, name=f'{name}_linear2')(x), [B, N, n_heads, dim_head])
        
        atn = jnp.einsum('bqhd, bkhd -> bhqk', q, k)
        atn /= np.sqrt(q.shape[-1])
        atn = jax.nn.softmax(atn, axis=-1)
        atn = hk.dropout(hk.next_rng_key(), dropout_rate, atn)
        
        out = jnp.einsum('bhqk, bkhd -> bqhd', atn, v)
        #out = einops.rearrange(out, 'b n h c -> b n (h c)')
        out = einops.rearrange(out, 'b q k d -> b q (k d)')
        out = hk.Linear(output_size = dim, name=f'{name}_linear3')(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0.0, out)
        return out, {}
    return attention_layer

def make_mixing_layer(dim, dim_inner, dropout_rate, name='mixing', **kwargs):
    def mixing_layer(x, training, check = None):
        out = hk.Linear(dim_inner,name=f'{name}_linear0')(x)
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0.0, out)
        out = hk.Linear(dim,name=f'{name}_linear1')(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0.0, out)
        return out, {}
    return mixing_layer

def make_film_generator(depth, dim, dim_inner, norm = None, activation = None, dropout = None, name = 'film_gen'):
    def film_generator(x, training):
        out = x
        for idx in range(depth - 1):
            out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = False,name=f'{name}_layer_norm{idx}')(out) if norm is not None else out
            out = hk.Linear(dim_inner,name=f'{name}_linear{idx}')(out)
            out = jax.nn.gelu(out) if activation is not None else out
            out = hk.dropout(hk.next_rng_key(), dropout if training else 0, out) if dropout is not None else out
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = False,name=f'{name}_layer_norm{depth}')(out) if norm is not None else out
        out = hk.Linear(dim*2,name=f'{name}_linear{depth}')(out)
        out = jax.nn.gelu(out) if activation is not None else out
        out = hk.dropout(hk.next_rng_key(), dropout if training else 0, out) if dropout is not None else out
        
        return jnp.split(out, 2, -1)
    return film_generator

def make_pos_cls_layer(dropout, name='pos_emb',**kwargs):
    def pos_cls_layer(x, training):
        B, N, C = x.shape
        cls_token = hk.get_parameter(f'{name}_cls_embedder', [1,   1, C], init = hk.initializers.RandomNormal() )
        pos_info  = hk.get_parameter(f'{name}_pos_embedder', [1, N+1, C], init = hk.initializers.RandomNormal(stddev=0.02) )
        
        cls_token = einops.repeat(cls_token, '() n d -> b n d', b = B)
        out = jnp.concatenate([cls_token, x], axis = 1)
        out += pos_info[:, :(N+1)]
        out = hk.dropout(hk.next_rng_key(), dropout if training else 0, out)
        
        return out
    return pos_cls_layer

def make_drop_path_rate(depth, drop_path_prob):
    if drop_path_prob is None:
        return [0.0]*depth
        
    if type(drop_path_prob) is float: 
        drop_path_prob = [0.0, drop_path_prob]
    
    if drop_path_prob[0] < 0.0 or drop_path_prob[0] >= 1.0 or drop_path_prob[1] < 0.0 or drop_path_prob[1] >= 1.0 :
        raise ValueError(f'drop_path_prob:{drop_path_prob} should be within 0.0 ~ 1.0')
    
    drop_path = []
    for i in range(depth):
        val = 1.0 - drop_path_prob[0] - i/(depth-1.0)*(drop_path_prob[1] - drop_path_prob[0])
        drop_path.append(1.0 - val)
    
    return drop_path

def drop_path(rng, rate, x, training = False, scale_by_keep = True):
    try:
        if rate < 0 or rate >= 1:
            raise ValueError(f"rate must be in [0, 1). {rate}")

        if rate == 0.0:
            return x
    except jax.errors.ConcretizationTypeError:
        pass
    if not training:
        return x
    
    shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
    keep_rate = 1.0 - rate
    keep = jax.random.bernoulli(rng, keep_rate, shape = shape)
    if keep_rate > 0.0 and scale_by_keep:
        jnp.divide(x, keep_rate)
    return keep * x 


















