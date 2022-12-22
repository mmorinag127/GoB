from copy import deepcopy
import haiku as hk
import jax.numpy as jnp
import functools

from .token_mixing import make_meta_layer, make_attention_layer, make_mixing_layer, make_mixer_layer, make_patch_layer, make_pos_cls_layer, make_drop_path_rate
from .moe_layer import make_MoE_layer_image, make_MoE_layer_token

def make_metaformer(model, patch_size, depth, dim, expansion, dropout, drop_path, layer_scale, **kwargs):
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
    if 'MoE' in kwargs.keys():
        moe_args = deepcopy(kwargs['MoE'])
        moe_cycle = moe_args.pop('cycle')
        moe_type = moe_args.pop('type')
        if moe_type == 'image':
            make_MoE_layer = make_MoE_layer_image
        elif moe_type == 'token':
            make_MoE_layer = make_MoE_layer_token
        else:
            raise ValueError(f'moe_type:{moe_type} is not supported!!')
    
    film_args = None
    if 'FiLM' in kwargs.keys():
        film_args = deepcopy(kwargs['FiLM'])
    
    def metaformer(x, gamma = None, beta = None, training = True, check = None):
        #dropout_rate = dropout if training else 0.0
        out = make_patch_layer(patch_size, dim, name='patch_layer')(x)
        if patch_emb_layer is not None:
            out = patch_emb_layer(name='patch_layer',**kwargs)(out, training)
        
        drop_path_rate = make_drop_path_rate(depth, drop_path)
        aux = {}
        
        for idx in range(depth):
            droppath_rate = drop_path_rate[idx]
            
            token_mixing = make_token_mixing_layer(dim, dim_inner, dropout, name=f'token_mixing{idx}',**kwargs)
            out, aux_ = make_meta_layer(dim, token_mixing, droppath_rate, layer_scale, name=f'token_mixing{idx}')(out, training, gamma = gamma, beta = beta)
            aux = update_aux(aux, aux_)
            
            chann_mixing = make_mixing_layer(dim, dim_inner, dropout, name=f'channel_mixing{idx}',**kwargs)
            if moe_cycle is not None and (idx+1) % moe_cycle == 0:
                chann_mixing = make_MoE_layer(chann_mixing, name = f'moe_layer{idx}', **moe_args)
            
            out, aux_ = make_meta_layer(dim, chann_mixing, droppath_rate, layer_scale, name=f'channel_mixing{idx}')(out, training, gamma = gamma, beta = beta)
            aux = update_aux(aux, aux_)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True, name='gap_layer_norm')(out)
        out = jnp.mean(out, axis=1)
        return out, aux
    return metaformer

def update_aux(this, other):
    for key, val in other.items():
        if key not in this.keys():
            this[key] = []
        this[key].append(val)
    return this

def make_head(model, depth, dim, expansion, dropout, n_classes, is_focal, **kwargs):
    
    dim_inner = dim*expansion
    
    def head(x, gamma = None, beta = None, training = True):
        dropout_rate = dropout if training else 0
        out = x
        
        for idx in range(depth):
            out, _ = make_meta_layer(dim, make_mixing_layer(dim, dim_inner, dropout_rate, name = f'head{idx}'), None, None, name = f'head{idx}')(out, training = training, gamma = gamma, beta = beta)
        
        out = hk.LayerNorm(axis=-1,create_scale=True,create_offset=True,name=f'head{depth}_layer_norm')(out)
        if is_focal:
            out = hk.Linear(output_size = n_classes, b_init = hk.initializers.Constant(-2.0), name=f'head{depth}_linear')(out)
        else:
            out = hk.Linear(output_size = n_classes, name=f'head{depth}_linear')(out)
        
        return out
    return head
