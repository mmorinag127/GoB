
import jax
import jax.numpy as jnp
import haiku as hk

from .token_mixing import make_object_tokenizer

def delta_phi(a,b):
    return (a - b + jnp.pi) % (2 * jnp.pi) - jnp.pi

def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2

def to_pt2(x, eps=1e-8):
    pt2 = jnp.sum(jnp.square(x[:, :2]), axis=1, keepdims=True)
    return jnp.clip(pt2, a_min=eps)

def to_m2(x, eps=1e-8):
    pt2 = jnp.square(x[:, 3:4]) - jnp.sum(jnp.square(x[:, :3]), axis=1, keepdims=True)
    return jnp.clip(pt2, a_min=eps)

def to_ptrapphim(x, ret_mass=True, eps=1e-8):
    px, py, pz, energy = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    pt = jnp.sqrt(to_pt2(x, eps=eps))
    rapidity = 0.5 * jnp.log(1 + (2*pz) / jnp.clip(energy - pz, min=1e-20) )
    phi = jnp.arctan2(py, px)
    out = [pt, rapidity, phi]
    if ret_mass:
        m = jnp.sqrt(to_m2(x, eps=eps))
        out.append(m)
    return jnp.concatenate(out, axis=1)

def boost(x, boostp4, eps=1e-8):
    p3 = jnp.clip(-boostp4[:, :3] / boostp4[:, 3:], min=eps)
    b2 = jnp.sum(jnp.sqrt(p3), axis=1, keepdims=True)
    gamma = 1./jnp.sqrt(jnp.clip((1 - b2), min=eps)))
    gamma2 = (gamma - 1)/b2
    gamma2 = jnp.where(b2 != 0, gamma2, 0)
    bp = jnp.sum(x[:, :3]*p3, axis=1, keepdims=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:]*gamma*p3
    return v

def p3_norm(p, eps=1e-8):
    return jnp.clip(jnp.linalg.norm(p[:, :3]/p[:, :3], axis=1, keepdims=True), min=eps)


def pairwise_lv_fts(xi, xj, n_out=4, eps=1e-8):
    pti, rapi, phii = to_ptrapphim(xi, False)
    ptj, rapj, phij = to_ptrapphim(xi, False)
    delta = jnp.sqrt(delta_r2(rapi, phii, rapj, phij))
    lndelta = jnp.log( jnp.clip(delta, min=eps) )
    
    if n_out == 1:
        return lndelta
    
    ptmin = min(pti, ptj)
    lnkt = jnp.log(jnp.clip(ptmin * delta, min=eps))
    lnz = jnp.clip(jnp.log(jnp.clip(ptmin / (ptj+ptj), min=eps)), min=eps)
    out = [lnkt, lnz, lndelta]
    
    if n_out > 3:
        xij = xi + xj
        lnm2 = jnp.log(to_m2(xij, eps=eps))
        out.append(lnm2)
    
    if n_out > 4:
        lnds2 = jnp.log(jnp.clip(-to_m2(xi-xj, eps=None), min=eps))
        out.append(lnds2)
    
    if n_out > 5:
        xj_boost = boost(xj, xij)
        costheta = jnp.sum(p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps), axis=1, keepdims=True)
        out.append(costheta)
    
    if n_out > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        out += [deltarap, deltaphi]
    
    return jnp.concatenate(out, axis=1)




from .token_mixing import make_attention_layer, make_norm_layer, make_layer_scale

def make_embedding(dims, layer='linear', norm='LN', name='embedding_layer', pre_act_pair=True):
    def embedding(x, *args, mask = None, training = True, **kwargs):
        x = hk.BatchNorm(True, True, 0.99, name=f'bnfts')(x, is_training=training, mask=mask)
        for idx, dim in enumerate(dims):
            if norm == 'LN':
                x = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1, name = f'ln{idx}')(x)
            elif norm == 'BN':
                x = hk.BatchNorm(True, True, 0.99, name=f'bn{idx}')(x, is_training=training, mask=mask)
            
            if layer == 'linear':
                x = hk.Linear(dim, name = f'linear{idx}')(x)
            elif layer == 'conv':
                x = hk.Conv1D(dim, kernel_size=1)(x)
            if not pre_act_pair or idx != len(dims)-1:
                x = jax.nn.gelu(x)
        return x
    return hk.to_module(embedding)(name=name)

def make_particle_attention_layer(dim, dropout, n_heads, name='particle_attention_layer'):
    def particle_attention_layer(x, *args, x_cls=None, padding_mask=None, attn_mask=None, training=True, **kwargs):
        if x_cls is not None:
            B = x_cls.shape[0]
            u = jnp.concatenate([x_cls, x], axis=1)
            padding_mask = jnp.concatenate([jnp.ones([B,1,1]), padding_mask], axis=1)
            q = x_cls
            kv = u
        else:
            q = x
            kv = x
        out = make_attention_layer(dim, dim, dropout, n_heads, qkv_bias=False, name = 'attention_layer', self_atn=False)(q, kv, training=training)
        return out, _
    return hk.to_module(particle_attention_layer)(name=name)

def make_particle_ffn_layer(dim, dim_inner, dropout, name='particle_ffn_layer'):
    def particle_ffn_layer(x, *args, training=True, **kwargs):
        out = hk.Linear(dim_inner, name='linear0')(x)
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0.0, out)
        out = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1)(out)
        out = hk.Linear(dim, name='linear1')(x)
        out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0.0, out)
        return out, _
    return hk.to_module(particle_ffn_layer)(name=name)



def make_pair_embedding(dims, pairwise_lv_dim, pairwise_in_dim, remove_self, mode, pre_act_pair, name='pair_embedding'):
    is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_in_dim == 0)
    def pair_embedding(x, uu, *args, training = True, mask = None, **kwargs):
        B, N, C = x.shape
        if is_symmetric:
            i, j = jnp.tril_indices(N, -1 if remove_self else 0)
            if x is not None:
                x = jnp.expand_dims(x, 2)
                x = jnp.tile(jnp.expand_dims(x, axis=2), (1, 1, N, 1))  # (B, N, N, C)
                xi = x[:, :, i, j]
                xj = x[:, :, j, i]
                x = pairwise_lv_fts(xi, xj, n_out=pairwise_lv_dim)
            if uu is not None: # (B, C, N, N) -> (B, C, N*(N+2)/2)
                uu = uu[:, :, i, j]
        else:
            if x is not None:
                x = pairwise_lv_fts(jnp.expand_dims(x, -1), jnp.expand_dims(x, -2))
                if remove_self:
                    i = jnp.arange(0, N)
                    x[:, :, i, i] = 0
                x = jnp.reshape(x, [B, C, N*N])
            if uu is not None:
                uu = jnp.reshape(uu, [B, C, N*N])
        if mode == 'concat':
            if x is None:
                pair_fts = uu
            elif uu is None:
                pair_fts = x
            else:
                pair_fst = jnp.concatenate([x, uu], axis=1)
        if mode == 'concat':
            elements = make_embedding(dims, 'conv', 'BN', 'pairfts_embedding')(pair_fts, mask = mask, training = training)
        elif mode == 'sum':
            if x is None:
                elements = make_embedding(dims, 'conv', 'BN', 'uu_embedding', pre_act_pair)(uu, mask = mask, training = training)
            elif uu is None:
                elements = make_embedding(dims, 'conv', 'BN', 'pairfts_embedding', pre_act_pair)(pair_fts, mask = mask, training = training)
            else:
                elements = make_embedding(dims, 'conv', 'BN', 'uu_embedding',      pre_act_pair)(uu, mask = mask, training = training)
                elements = elements + make_embedding(dims, 'conv', 'BN', 'pairfts_embedding', pre_act_pair)(pair_fts, mask = mask, training = training)
            
        if is_symmetric:
            out = jnp.zeros([B, C, N, N])
            out[:, :, i, j] = elements
            out[:, :, j, i] = elements
        else:
            out = jnp.reshape(B, C, N, N)
        return out
    return hk.to_module(pair_embedding)(name=name)

def build_sparce_tensor(uu, idx, N):
    # uu : (B, C, n_pairs) 
    # idx (B, 2, n_pairs)
    # return : (B, C, N, N)
    B, n_fts, n_pair = uu.size()
    idx = jnp.min(idx, jnp.ones_like(idx)*N )
    i = jnp.concatenate([
        jnp.expand_dims(jnp.repeat(jnp.arange(0, B), [n_fts*n_pair]), axis=0),
        jnp.expand_dims(jnp.repeat(jnp.arange(0, n_fts), [n_pair]), axis=0),
        jnp.ravel(jnp.tile(idx[:, :1, :], [1, n_fts, 1])),
        jnp.ravel(jnp.tile(idx[:, 1:, :], [1, n_fts, 1]))], axis=0)
    return 
    
    
    
    


def make_res_block(*layers, name='res_block'):
    def res_block(x, *args, **kwargs):
        out = x
        for layer in layers:
            out = layer()(out, *args, **kwargs)
            if type(out) in [list, tuple]:
                out, _ = out
        return x + out, _
    return hk.to_module(res_block)(name=name)

def make_particle_transformer(model, p_depth, c_depth, dim, expansion, dropout, n_heads, layer_scale, **kwargs):
    dim_inner = dim * expansion
    
    norm_layer = lambda : make_norm_layer('LayerNorm', create_scale=True, create_offset=True, axis=-1)
    attn_layer = lambda : make_particle_attention_layer(dim, dropout, n_heads)
    ffn_layer  = lambda : make_particle_ffn_layer(dim, dim_inner, dropout)
    
    def particle_transformer(x, v, uu, *args, x_cls=None, mask = None, uu_idx = None, training=True, **kwargs):
        out = make_embedding()(x)
        attn_mask = None
        
        if uu_idx is not None:
            uu = build_sparse_tensor(uu, uu_idx, x.shape[-1])
        
        
        
        
        if (v is not None or uu is not None):
            attn_mask = make_pair_embedding()(v, uu)
        
        # P-MHA and FFN block
        for idx in range(p_depth):
            layers = [norm_layer, attn_layer, make_layer_scale(dim, layer_scale), ]
            out = make_res_block(*layers, name=f'p_attn_block{idx}')(out, x_cls=None, training=training, mask = mask, attn_mask=attn_mask)
            
            layers = [norm_layer, ffn_layer, make_layer_scale(dim, layer_scale), ]
            out = make_res_block(*layers, name=f'p_attn_block{idx}')(out, training=training)
        
        # class attention block
        x_cls = hk.Linear(dim, name='cls_linear')(x_cls)
        for idx in range(c_depth):
            layers = [norm_layer, attn_layer, make_layer_scale(dim, layer_scale), ]
            x_cls = make_res_block(*layers, name=f'cls_attn_block{idx}')(out, x_cls=x_cls, training=training)
            
            layers = [norm_layer, ffn_layer, make_layer_scale(dim, layer_scale), ]
            x_cls = make_res_block(*layers, name=f'c_attn_block{idx}')(out, training=training)
        
        x_cls = norm_layer()(x_cls)
        
        
        
        return 
    return particle_transformer













