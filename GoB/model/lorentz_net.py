
import jax
import jax.numpy as jnp
import haiku as hk


def normsq4(p):
    # Minkowski square norm
    # `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    psq = jnp.power(p, 2)
    return 2 * psq[..., 0] - jnp.sum(psq, axis=-1)

def dotsq4(p,q):
    # Minkowski inner product
    #     `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    psq = p*q
    return 2 * psq[..., 0] - jnp.sum(psq, axis=-1)

def psi(p):
    # `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    return jnp.sign(p) * jnp.log(jnp.abs(p) + 1)

def unsorted_segment_sum(data, segment_ids, num_segments):
    # Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    # Adapted from https://github.com/vgsatorras/egnn.
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result



def make_LGEB(dim, dim_inner, c_weight = 1.0, last_layer=False, name='LGEB'):
    def LGEB(h, x, edges, node_attr):
        ### minkowski_feats
        i, j = edges
        x_diff = x[i] - x[j]
        norms = psi(jnp.unsqueeze(normsq4(x_diff), axis=1))
        dots = psi(jnp.unsqueeze(dotsq4(x[i], x[j]), axis=1))
        
        ### m_model
        # phi_3\e
        m = jnp.concatenate([h[i], h[j], norms, dots], axis=1)
        m = hk.Linear(dim_inner, with_bias=False, name=f'm_phi_e_linear1')(m)
        m = hk.BatchNorm(True, True, 0.99, name=f'm_phi_e_bn')(m)
        m = jax.nn.relu(m)
        m = hk.Linear(dim_inner, name=f'm_phi_e_linear2')(m)
        m = jax.nn.relu(m)
        # phi_m
        w = hk.Linear(dim_inner, with_bias=False, name=f'm_phi_m_linear1')(m)
        w = jax.nn.sigmoid(w)
        m = m * w
        
        ### x_model
        if not last_layer:
            # phi_x 
            phi_x = hk.Linear(dim_inner, with_bias=False, name=f'x_linear1')(m)
            phi_x = jax.nn.relu(phi_x)
            phi_x = hk.Linear(1, with_bias=False, name=f'x_linear2')(phi_x)
            trans = x_diff * phi_x
            trans = jnp.clip(trans, min=-100., max=100.)
            xagg = unsorted_segment_sum(trans, i, x.shape[0])
            x = x + xagg * c_weight
        
        ### h_model
        h_agg = unsorted_segment_sum(m, i, num_segments=h.shape[0])
        h_agg = jnp.concatenate([h, h_agg, node_attr], axis=1)
        # phi_h
        h_agg = hk.Linear(dim_inner, with_bias=False, name=f'h_linear1')(h_agg)
        h_agg = hk.BatchNorm(True, True, 0.99, name=f'h_bn')(h_agg)
        h_agg = jax.nn.relu(h_agg)
        h_agg = hk.Linear(dim, name=f'h_linear2')(h_agg)
        h = h + h_agg
        return h, x, m
    return hk.to_module(LGEB)(name=name)



from .token_mixing import make_object_tokenizer

def make_lorentz_net(model, depth, dim, expansion, c_weight, **kwargs):
    dim_inner = dim * expansion
    def lorentz_net(glob, x, edges, mask, one_hot, *args, training = True, **kwargs):
        
        feature, mask = make_object_tokenizer(dim, w_init=None, name='tokenizer')(feature, one_hot, mask = mask)
        
        big_pos = jnp.finfo(feature.dtype).max
        # position = jnp.where(mask, position, big_pos)
        h = hk.Linear(dim_inner, name = 'glob_embedding')(glob)
        for idx in range(depth):
            h, x, _ = make_LGEB(dim_inner, dim_inner, c_weight=c_weight, last_layer=idx==depth-1)(h, x, edges, node_attr=glob)
        
        h = h * node_mask
        
        
        return 
    return lorentz_net
