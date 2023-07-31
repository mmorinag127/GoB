import haiku as hk
import jax
import jax.numpy as jnp
# import numpy as np
import einops
from .weight_init import make_w_init

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

def make_patch_layer(patch_size, dim, w_init = None, name = 'patch_emb'):
    w_init = make_w_init(w_init)
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
        patch = hk.Linear(dim, w_init = w_init, name = f'linear')(patch)
        #patch = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = False, name = f'{name}_layer_norm')(patch)
        return patch
    return hk.to_module(patch_layer)(name=name)

def make_layer_norm(axis, scale, offset, name='layer_norm'):
    def layer_norm(x, mask = None):
        ln = hk.LayerNorm(axis = axis, create_scale = scale, create_offset = offset, name=name)
        out = ln(x, mask = None)
        # if mask is not None:
        #     out = jnp.where(mask, out, 0)
        return out
    return layer_norm

def make_tokenizer_layer(dim, w_init = None, name = 'tokenizer'):
    w_init = make_w_init(w_init)
    def tokenizer_layer(x, *args, mask = None, **kwargs):
        token = hk.Linear(dim, w_init = w_init, name = f'linear')(x)
        # if mask is not None:
        #     dtype = token.dtype
        #     big_neg = jnp.finfo(dtype).min
        #     token = jnp.where(mask, token, big_neg)
        return token
    return hk.to_module(tokenizer_layer)(name=name)

def make_object_tokenizer(dim, w_init = None, name = 'object_tokenizer'):
    w_init = make_w_init(w_init)
    def object_tokenizer(x, *args, one_hot = None, glob = None, mask = None, **kwargs):
        tokens = hk.Linear(dim, w_init = w_init, name = f'x_linear')(x)
        if one_hot is not None:
            id_emb = hk.Linear(dim*2, w_init = w_init, name = f'id_linear')(one_hot)
            id_beta, id_gamma = jnp.split(id_emb, 2, -1)
            id_beta  = jax.nn.gelu(id_beta)
            id_gamma = jax.nn.gelu(id_gamma)
            tokens = id_gamma * tokens + id_beta
        
        if glob is not None:
            glob = hk.Linear(dim, w_init=w_init, name = f'glob_linear')(glob)
            B,D = glob.shape
            glob = jnp.reshape(glob, [B,1,D])
            tokens = jnp.concatenate([glob, tokens[:, :-1, :]], axis=1)
            if mask is not None:
                m = jnp.ones([B, 1, 1], dtype=mask.dtype)
                mask = jnp.concatenate([m, mask[:,:-1]], axis=1)
        return tokens, mask
    return hk.to_module(object_tokenizer)(name=name)

def make_meta_layer(dim, mixing_layer, norm_layer, drop_path_rate, lscale, w_init = None, name = 'meta'):
    w_init = make_w_init(w_init)
    def meta_layer(x, *args, training = True, gamma = None, beta = None, mask = None, **kwargs):
        #out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = create_s_o, create_offset = create_s_o, name=f'{name}_layer_norm')(x, mask=_mask)
        #out = norm_layer()(x, mask = mask)
        out = norm_layer()(x)
        
        out, aux = mixing_layer()(out, *args, training = training, mask = mask)
        if gamma is not None and beta is not None:
            out = make_film_module(dim, w_init)(out, training, gamma, beta)
        
        if lscale is not None and lscale > 0.0:
            param = hk.get_parameter(name=name, shape = [dim], init=hk.initializers.Constant(constant = lscale))
            out = param*out
        
        if drop_path_rate is not None:
            out = drop_path(drop_path_rate if training else 0.0, out)
        
        return x + out, aux
    return hk.to_module(meta_layer)(name=name)

def make_layer_scale(dim, lscale, name = 'layer_scale'):
    def layer_scale(x, *args, **kwargs):
        shape = [dim]
        if dim is None:
            shape = [x.shape[-1]]
        param = hk.get_parameter(name=name, shape = shape, init=hk.initializers.Constant(constant = lscale))
        out = param*x
        return out
    return hk.to_module(layer_scale)(name=name)

def make_film_module(dim, w_init, name='film'):
    def film_module(x, *args, gamma=None, beta = None, **kwargs):
        gamma = hk.Linear(dim, w_init = w_init, name=f'gamma')(gamma)
        beta  = hk.Linear(dim, w_init = w_init, name=f'beta')(beta)
        gamma = jax.nn.gelu(gamma)
        beta  = jax.nn.gelu(beta)
        if x.ndim > 2:
            gamma = jnp.reshape(gamma, (gamma.shape[0], -1, gamma.shape[-1]))
            beta  = jnp.reshape(beta,  (beta.shape[0],  -1, beta.shape[-1]))
        out = gamma*x + beta
        return out
    return hk.to_module(film_module)(name=name)

def make_apply_mask(value):
    def apply_mask(x, *args, mask = None, **kwargs):
        if mask is not None:
            out = jnp.where(mask, x, value)
            return out
        return x
    return hk.to_module(apply_mask)

def make_norm_layer(model, type, *args, name = 'norm', **kwargs):
    if hasattr(hk, type):
        def norm_layer(x, *xargs, **xkwargs):
            return getattr(hk, type)(*args, **kwargs)(x)
        return hk.to_module(norm_layer)()
    elif type in ['GRN', 'grn']:
        return make_global_responce_norm(*args, **kwargs)
    raise ValueError(f'{type} is not supported in make_norm_layer')

def make_global_responce_norm(reduce=(1,), eps=1.0e-6, name = 'grn'):
    # https://arxiv.org/pdf/2301.00808.pdf
    def global_responce_norm(x, *args, mask=None, **kwargs):
        _, _, dim = x.shape
        gamma = hk.get_parameter(name=f'gamma', shape = [1, dim], init=hk.initializers.Constant(constant = 0.0))
        beta  = hk.get_parameter(name=f'beta',  shape = [1, dim], init=hk.initializers.Constant(constant = 0.0))
        if mask is not None:
            mask_expd = einops.repeat(mask==1, 'b n () -> b n d', d = dim)
            x = jnp.where(mask_expd, x, 0.0)
        
        # gx = torch.norm(X, p=2, dim=(1,2), keepdim=True)
        gx = jnp.linalg.norm(x, ord=2, axis=reduce, keepdims=True)
        nx = gx / (jnp.mean(gx, axis=-1, keepdims=True)+eps)
        return gamma * (x * nx) + beta + x
    return hk.to_module(global_responce_norm)(name=name)

def make_mixer_layer(dim, dim_inner, dropout_rate, n_heads, w_init = None, name = 'mixer',**kwargs):
    w_init = make_w_init(w_init)
    def mixer_layer(x, *args, training = True, **kwargs):
        B, N, _ = x.shape
        dim_head = dim_inner // n_heads
        out = jnp.swapaxes(x, 1, 2)
        out = hk.Linear(dim_inner, w_init = w_init, name=f'linear0')(out)
        out = jnp.reshape(out, [B, dim, n_heads, dim_head])
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout_rate if training else 0.0, out)
        out = hk.Linear(N//n_heads, w_init = w_init, name=f'linear1')(out)
        out = jnp.reshape(out, [B, dim, N] )
        out = jnp.swapaxes(out, 1, 2)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout_rate if training else 0.0, out)
        return out, {}
    return hk.to_module(mixer_layer)(name=name)

def make_attention_layer(dim, dim_inner, dropout, n_heads, qkv_bias, w_init = None, name = 'self_attention', self_atn=True, **kwargs):
    w_init = make_w_init(w_init)
    dim_head = dim_inner//n_heads
    def attention_layer(x, y, *args, training = False, mask = None, **kwargs):
        B, _, _ = x.shape
        q = jnp.reshape(hk.Linear(dim_inner, w_init = w_init, with_bias = qkv_bias, name=f'linear_q')(x), [B, -1, n_heads, dim_head])
        k = jnp.reshape(hk.Linear(dim_inner, w_init = w_init, with_bias = qkv_bias, name=f'linear_k')(y), [B, -1, n_heads, dim_head])
        v = jnp.reshape(hk.Linear(dim_inner, w_init = w_init, with_bias = qkv_bias, name=f'linear_v')(y), [B, -1, n_heads, dim_head])
        
        atn = jnp.einsum('bqhd, bkhd -> bhqk', q, k)
        #atn /= np.sqrt(q.shape[-1]) # numpy function would have float64 as a default dtype
        atn /= jnp.sqrt(dim_head)
        if mask is not None:
            dtype = q.dtype
            big_neg = jnp.finfo(dtype).min
            _mask = jnp.expand_dims(mask, 1)
            _mask = einops.repeat(_mask, 'b () n () -> b h n d', h = atn.shape[1], d = atn.shape[-1])
            atn = jnp.where(_mask, atn, big_neg)
        
        atn = jax.nn.softmax(atn, axis=-1)
        atn = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0.0, atn)
        
        out = jnp.einsum('bhqk, bkhd -> bqhd', atn, v)
        #out = einops.rearrange(out, 'b n h c -> b n (h c)')
        out = einops.rearrange(out, 'b q h d -> b q (h d)')
        
        out = hk.Linear(dim, w_init = w_init, name=f'linear')(out)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0.0, out)
        return out, {}
    
    def self_attention_layer(x, *args, training = False, mask = None, **kwargs):
        return attention_layer(x, x, training = training, mask = mask)
    
    if self_atn:
        return hk.to_module(self_attention_layer)(name=name)
    return hk.to_module(attention_layer)(name=name)

def make_mixing_layer(dim, dim_inner, dropout_rate, w_init = None, name='mixing', **kwargs):
    w_init = make_w_init(w_init)
    def mixing_layer(x, *args, training = True, **kwargs):
        out = hk.Linear(dim_inner, w_init = w_init, name=f'linear0')(x)
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout_rate if training else 0.0, out)
        out = hk.Linear(dim, w_init = w_init, name=f'linear1')(out)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout_rate if training else 0.0, out)
        return out, {}
    return hk.to_module(mixing_layer)(name=name)

def make_ffn_layer(dim, dim_inner, dropout_rate, w_init = None, name='ffn', with_aux = True, **kwargs):
    w_init = make_w_init(w_init)
    def ffn_layer(x, *args, training = False, **kwargs):
        B, N, _ = x.shape
        out = hk.Linear(dim_inner, w_init = w_init, name=f'linear0')(x)
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout_rate if training else 0.0, out)
        out = hk.Linear(dim, w_init = w_init, name=f'linear1')(out)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout_rate if training else 0.0, out)
        if with_aux:
            return out, {}
        return out
    return hk.to_module(ffn_layer)(name=name)

def make_film_gen(model, depth, dim, expansion, w_init = None, norm = False, activation = True, dropout = 0.0, name = 'film_gen', vkey = 'glob'):
    dim_inner = dim*expansion
    w_init = make_w_init(w_init)
    def film_gen(x, *args, training = False, **kwargs):
        out = x[vkey]
        
        
        for idx in range(depth - 1):
            if norm:
                out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = False,name=f'ln{idx}')(out)
            out = hk.Linear(dim_inner, w_init = w_init, name=f'linear{idx}')(out)
            if activation:
                out = jax.nn.gelu(out)
            if dropout > 0.0:
                out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0, out)
        
        if norm:
            out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = False, name=f'ln{depth}')(out)
        
        out = hk.Linear(dim*2, w_init = w_init, name=f'linear{depth}')(out)
        if activation:
            out = jax.nn.gelu(out)
        if dropout > 0.0:
            out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0, out)
        return jnp.split(out, 2, -1)
    return hk.to_module(film_gen)(name = name)

def make_pos_cls_layer(dropout, name='pos_emb',**kwargs):
    def pos_cls_layer(x, *args, training = False, **kwargs):
        B, N, C = x.shape
        
        cls_token = hk.get_parameter(f'cls_embed', [1,   1, C], init = hk.initializers.RandomNormal(), dtype = x.dtype )
        pos_info  = hk.get_parameter(f'pos_embed', [1, N+1, C], init = hk.initializers.RandomNormal(stddev=0.02), dtype = x.dtype )
        cls_token = einops.repeat(cls_token, '() n d -> b n d', b = B)
        out = jnp.concatenate([cls_token, x], axis = 1)
        out += pos_info[:, :(N+1)]
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0, out)
        return out
    return hk.module(pos_cls_layer)(name=name)

def make_drop_path_rate(depth, drop_path_prob):
    if drop_path_prob is None or drop_path_prob == 0:
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

def make_drop_path(rate, name = 'drop', scale_by_keep = True):
    try:
        if rate < 0 or rate >= 1:
            raise ValueError(f"rate must be in [0, 1). {rate}")
        if rate == 0.0:
            def drop_path(x, *args, **kwargs):
                return x
            return drop_path
    except jax.errors.ConcretizationTypeError:
        pass
    
    def drop_path(x, *args, training = True, **kwargs):
        if not training:
            return x
        
        shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
        keep_rate = 1.0 - rate
        rng = hk.next_rng_key() if hk.running_init() else None
        keep = jax.random.bernoulli(rng, keep_rate, shape = shape)
        if keep_rate > 0.0 and scale_by_keep:
            jnp.divide(x, keep_rate)
        return keep * x 
    return hk.to_module(drop_path)(name=name)

def make_token_merging_layer(merging_ratio, name = 'token_merging'):
    def token_merging_layer(x, training = True, gamma = None, beta = None, mask = None):
        B, n_tokens, dim = x.shape
        
        
        
        return
    return token_merging_layer





























































