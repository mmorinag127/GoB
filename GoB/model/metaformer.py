import haiku as hk
import jax.numpy as jnp
import functools

from .token_mixing import make_meta_layer, make_meta_layer_with, make_attention_layer, make_mixing_layer, make_mixer_layer, make_patch_layer, make_pos_cls_layer, make_drop_path_rate
from .moe_layer import make_moe_layer

def make_metaformer(model, patch_size, depth, dim, expansion, dropout, drop_path, layer_scale, n_classes, moe_cycle, **kwargs):
    dim_inner = dim*expansion
    patch_emb_layer = None
    
    if 'ViT' in model:
        make_token_mixing_layer = make_attention_layer
        patch_emb_layer = functools.partial(make_pos_cls_layer, dropout = dropout)
    elif 'Mixer' in model:
        make_token_mixing_layer = make_mixer_layer
    else:
        raise ValueError(f'model : {model} is not supported!!')
    
    def metaformer(x, training, check = None):
        dropout_rate = dropout if training else 0
        out = make_patch_layer(patch_size, dim)(x)
        if patch_emb_layer is not None:
            out = patch_emb_layer(**kwargs)(out, training)
        
        drop_path_rate = make_drop_path_rate(depth, drop_path)
        total_aux_loss = 0.
        expert_weights = []
        for idx in range(depth):
            droppath_rate = drop_path_rate[idx] if training else 0.0
            out = make_meta_layer(dim, make_token_mixing_layer(dim, dim_inner, dropout_rate, **kwargs), droppath_rate, layer_scale)(out, training)
            if moe_cycle is not None and (idx+1) % moe_cycle == 0:
                if check is None:
                    out, aux_loss = make_meta_layer(dim, make_moe_layer(dim, dim_inner, dropout_rate = dropout_rate, **kwargs), droppath_rate, layer_scale, hax_aux = True)(out, training)
                    total_aux_loss += aux_loss
                else:
                    out, aux_loss, expert_weight = make_meta_layer_with(dim, make_moe_layer(dim, dim_inner, dropout_rate = dropout_rate, **kwargs), droppath_rate, layer_scale, hax_aux = True)(out, training, check = True)
                    total_aux_loss += aux_loss
                    expert_weights.append(expert_weight)
            else:
                out = make_meta_layer(dim, make_mixing_layer(dim, dim_inner, dropout_rate, **kwargs), droppath_rate, layer_scale)(out, training)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        out = jnp.mean(out, axis=1)
        out = hk.Linear(output_size = n_classes)(out)
        if len(expert_weights) > 0:
            expert_weights = jnp.stack(expert_weights, axis = 0)
            return out, total_aux_loss, expert_weights
        return out, total_aux_loss
    return metaformer










