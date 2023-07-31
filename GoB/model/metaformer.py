from copy import deepcopy
import haiku as hk
import jax.numpy as jnp
import functools
import einops

from .token_mixing import make_meta_layer, make_attention_layer, make_mixing_layer, make_mixer_layer, make_patch_layer, make_pos_cls_layer, make_drop_path_rate, make_ffn_layer
from .moe_layer import make_moe_block
# from .weight_init import make_w_init
from .token_mixing import make_tokenizer_layer, make_norm_layer, make_layer_scale, make_drop_path, make_film_module, make_object_tokenizer

def make_metaformer(model, patch_size, depth, dim, expansion, dropout, drop_path, layer_scale, w_init, moe_args, **kwargs):
    dim_inner = dim*expansion
    patch_emb_layer = None
    
    if 'ViT' in model:
        make_token_mixing_layer = make_attention_layer
        patch_emb_layer = functools.partial(make_pos_cls_layer, dropout = dropout)
    elif 'Mixer' in model:
        make_token_mixing_layer = make_mixer_layer
    else:
        raise ValueError(f'model : {model} is not supported!!')
    
    moe_cycle = None
    if moe_args is not None:
        moe_cycle = moe_args.pop('cycle')
        moe_type = moe_args.pop('type')
        if moe_type == 'SMoE':
            make_MoE_layer = make_sparse_noisy_topk_gating
        else:
            raise ValueError(f'moe_type:{moe_type} is not supported!!')
    
    drop_path_rate = make_drop_path_rate(depth, drop_path)
    def metaformer(x, gamma = None, beta = None, training = True, check = None):
        #dropout_rate = dropout if training else 0.0
        out = make_patch_layer(patch_size, dim, w_init, name='patch_layer')(x)
        if patch_emb_layer is not None:
            out = patch_emb_layer(name='patch_layer',**kwargs)(out, training)
        aux = {}
        
        for idx in range(depth):
            droppath_rate = drop_path_rate[idx]
            
            token_mixing = make_token_mixing_layer(dim, dim_inner, dropout, w_init = w_init, name=f'token_mixing{idx}',**kwargs)
            out, aux_ = make_meta_layer(dim, token_mixing, droppath_rate, layer_scale, w_init = w_init, name=f'token_mixing{idx}')(out, training, gamma = gamma, beta = beta)
            aux = update_aux(aux, aux_)
            
            chann_mixing = make_mixing_layer(dim, dim_inner, dropout, w_init = w_init, name=f'channel_mixing{idx}',**kwargs)
            if moe_cycle is not None and (idx+1) % moe_cycle == 0:
                chann_mixing = make_MoE_layer(chann_mixing, name = f'moe_layer{idx}', **moe_args)
            
            out, aux_ = make_meta_layer(dim, chann_mixing, droppath_rate, layer_scale, w_init = w_init, name=f'channel_mixing{idx}')(out, training, gamma = gamma, beta = beta)
            aux = update_aux(aux, aux_)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True, name='gap_layer_norm')(out)
        out = jnp.mean(out, axis=1)
        return out, aux
    return metaformer

def make_transformer(model, depth, dim, expansion, dropout, drop_path, layer_scale, w_init, moe_args, norm_args, **kwargs):
    dim_inner = dim*expansion
    
    token_mixing = make_token_mixing_layer(model, dim=dim, dim_inner=dim_inner, dropout=dropout, w_init=w_init, name='token_mixing', **kwargs)
    ffn_mixing = lambda : make_ffn_layer(dim, dim_inner, dropout, w_init = w_init, name=f'channel_mixing', **kwargs)
    moe_layer  = make_moe_layer(**moe_args)
    norm_layer = lambda : make_norm_layer(**norm_args)
    
    drop_path_rate = make_drop_path_rate(depth, drop_path)
    def transformer(x, mask, gamma = None, beta = None, training = True):
        #dropout_rate = dropout if training else 0.0
        #mask = jnp.expand_dims(mask == 1, -1)
        mask = mask == 1
        out = make_tokenizer_layer(dim, w_init, name='tokenizer')(x, mask = mask)
        
        aux = {}
        mask_expd = einops.repeat(mask==1, 'b n () -> b n d', d = dim)
        for idx in range(depth):
            droppath_rate = drop_path_rate[idx]
            
            token_mixing_layer = make_meta_layer(dim, token_mixing, norm_layer, droppath_rate, layer_scale, w_init = w_init, name=f'token_block{idx}')
            out, aux = update_outputs(aux, *token_mixing_layer(out, training = training, gamma = gamma, beta = beta, mask = mask), mask_expd)
            
            moe_ffn_mixing = moe_layer(ffn_mixing, idx, name='moe')
            ch_mixig_layer = make_meta_layer(dim, moe_ffn_mixing, norm_layer, droppath_rate, layer_scale, w_init = w_init, name=f'channel_block{idx}')
            out, aux = update_outputs(aux, *ch_mixig_layer(out, training = training, gamma = gamma, beta = beta, mask = mask), mask_expd)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = True, name='ln')(out)
        #out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True, name='ln')(out)
        # out = make_layer_norm(-1, True, False, name='gap_ln')(out, mask = mask_expd)
        out = jnp.mean(out, axis=1, where=mask)
        return out, aux
    return transformer

def make_token_mixing_layer(type, **kwargs):
    if 'FvT' in type:
        token_mixing = lambda : make_attention_layer(**kwargs)
    elif 'GoB' in type:
        token_mixing = lambda : make_attention_layer(**kwargs)
    elif 'Mixer' in type:
        token_mixing = lambda : make_mixer_layer(**kwargs)
    else:
        raise ValueError(f'model : {type} is not supported!!')
    return token_mixing

def make_go_beyond(model, depth, dim, expansion, dropout, drop_path, layer_scale, w_init, norm_args, 
                   moe_args = None, film_args = None, **kwargs):
    dim_inner = dim*expansion
    moe_args = moe_args if type(moe_args) is dict else {'moe_type':None}
    
    token_mixing = make_token_mixing_layer(model, dim=dim, dim_inner=dim_inner, dropout=dropout, w_init=w_init, name='token_mixing', **kwargs)
    ffn_mixing = lambda : make_ffn_layer(dim, dim_inner, dropout, w_init = w_init, name=f'channel_mixing', with_aux = False, **kwargs)
    norm_layer = lambda : make_norm_layer(**norm_args)
    
    make_moe_layer  = make_moe_block(moe_args)
    drop_path_rate = make_drop_path_rate(depth, drop_path)
    
    def go_beyond(feature, mask, position = None, one_hot = None, glob = None, gamma=None, beta=None, training=True):
        tokens, mask = make_object_tokenizer(dim, w_init, name='tokenizer')(feature, one_hot = one_hot, glob = glob, mask = mask)
        if position is not None:
            pos_emb = hk.Linear(dim, name = 'position')(position)
            tokens = tokens + pos_emb
        
        if mask is not None:
            if len(mask.shape) == 2:
                B,N = mask.shape
                mask = jnp.reshape(mask, [B, N, 1])
        mask = mask == 1
        mask_expd = einops.repeat(mask, 'b n () -> b n d', d = dim)
        tokens = apply_mask(tokens, mask_expd)
        aux = {}
        
        for idx in range(depth):
            # token mixing lyaer
            layers = [norm_layer, token_mixing]
            if layer_scale is not None and layer_scale > 0.0:
                layers.append(lambda : make_layer_scale(dim, layer_scale))
            
            if drop_path_rate[idx] > 0.0:
                layers.append(lambda : make_drop_path(drop_path_rate[idx], name='drop_path'))
            
            tokens, _aux = make_res_block(*layers, name='token_block')(tokens, mask=mask, gamma=gamma, beta=beta, training=training)
            aux = update_aux(aux, _aux)
            tokens = apply_mask(tokens, mask_expd)
            
            # ffn layer
            moe_layer = make_moe_layer(ffn_mixing, idx, name='moe')
            layers = [norm_layer, moe_layer]
            if gamma is not None and beta is not None:
                layers.append(lambda : make_film_module(dim, w_init, name='film'))
                
            if layer_scale is not None and layer_scale > 0.0:
                layers.append(lambda : make_layer_scale(dim, layer_scale))
                
            if drop_path_rate[idx] > 0.0:
                layers.append(lambda : make_drop_path(drop_path_rate[idx], name='drop_path'))
            
            tokens, aux = make_res_block(*layers, name='ffn_block')(tokens, mask=mask, gamma=gamma, beta=beta, training=training)
            aux = update_aux(aux, _aux)
            tokens = apply_mask(tokens, mask_expd)
        
        tokens = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = True, name='ln')(tokens)
        # out = jnp.mean(out, axis=1, where=mask)
        return tokens, aux
    return go_beyond

def apply_mask(x, mask, value=0):
    if mask is not None:
        return jnp.where(mask, x, value)
    return x

def make_res_block(*layers, name='res_block'):
    def res_block(x, *args, **kwargs):
        aux = {}
        out = x
        for layer in layers:
            out = layer()(out, *args, **kwargs)
            if type(out) in [list, tuple]:
                out, _aux = out
                aux = update_aux(aux, _aux)
        return x + out, aux
    return hk.to_module(res_block)(name=name)

def update_aux(aux, aux_):
    if aux_ is None:
        return aux
    for key, val in aux_.items():
        if key not in aux.keys():
            aux[key] = []
        aux[key].append(val)
    return aux

def update_outputs(aux, out_, aux_, mask = None):
    aux = update_aux(aux, aux_)
    if mask is not None:
        out = jnp.where(mask, out_, 0)
    return out, aux
class Test(hk.Module):
    def  __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
    def __call__(self, batch, training = True, check = None):
        #rng = hk.PRNGSequence(rng)
        image = batch['image']
        x = jnp.reshape(image, (image.shape[0], -1))
        out = hk.Linear(self.n_class*10)(x)
        out = hk.Linear(self.n_class*10)(out)
        
        hk.dropout(hk.next_rng_key() if hk.running_init() else None, 0.1 if training else 0.0, out)
        
        out = hk.Linear(self.n_class)(out)
        return out, {}
