
import jax
import jax.numpy as jnp
import haiku as hk

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


def make_edge_conv(dim, K, in_depth, name='edge_conv', eps=1e-7):
    def edge_conv(position, feature, *args, training = True, mask=None, **kwargs):
        # distance
        # r_A = jnp.sum(position, axis=-1, keepdims=True)
        # m = jnp.einsum('BNC,BMC -> BNM', position, position)
        # D = r_A - 2.0 * m + jnp.transpose(r_A, axes = (0,2,1))
        
        
        D = jnp.sum((position[:, None, :] - position[:, :, None]) ** 2, axis=-1)
        
        N = feature.shape[1]
        
        # # knn
        indices = naive_top_k(-D, k = K+1, axis = -1)[:, :, 1:]
        indices = jax.nn.one_hot(indices, num_classes=N, axis=-1, dtype=jnp.int64)
        knn_fts = jnp.einsum('BNC,BMKN -> BMKC', feature, indices)
        knn_fts_center = jnp.tile(jnp.expand_dims(feature, axis=2), (1, 1, K, 1))  # (N, P, K, C)
        knn_fts = jnp.concatenate([knn_fts_center, jnp.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)
        
        
        mask2d = jnp.tile(jnp.expand_dims(mask, axis=2), (1,1,K,1))
        for idx in range(in_depth):
            knn_fts = hk.BatchNorm(True, True, 0.99, name=f'bn{idx}')(knn_fts, is_training=training, mask=mask2d)
            knn_fts = hk.Conv2D(dim, kernel_shape = (1,1), stride=1, name=f'conv{idx}', with_bias=False, data_format='NHWC', padding='VALID')(knn_fts)
            knn_fts = jax.nn.relu(knn_fts)
        
        
        knn_fts = jnp.mean(knn_fts, axis = 2)
        #xf = jnp.expand_dims(feature, axis=2
        xf = feature
        xf = hk.BatchNorm(True, True, 0.99, name=f'bnxf')(xf, is_training=training, mask=mask)
        #xf = hk.Conv2D(dim, kernel_shape = (1,1), stride=1, with_bias=False, name=f'convxf', data_format='NHWC', padding='VALID')(xf)
        xf = hk.Conv1D(dim, kernel_shape = 1, stride=1, with_bias=False, name=f'convxf', data_format='NWC', padding='VALID')(xf)
        #xf = hk.Linear(dim, with_bias=False, name=f'linear_xf')(xf)
        
        #xf = jnp.squeeze(xf, axis=2)
        out = jax.nn.relu(xf + knn_fts)
        # out = jax.nn.relu(xf)
        return out
    return hk.to_module(edge_conv)(name=name)


from .token_mixing import make_object_tokenizer

def logPt(v):
    return jnp.where(v>0., jnp.log(v), 0.0)


def make_particle_net(model, depth, dim, expansion, K, in_depth, **kwargs):
    def particle_net(feature, position,  mask, *args, one_hot = None, training = True, **kwargs):
        # norm input
        # feature = feature.at[:,:,0].set(logPt(feature[:,:,0]))
        if one_hot is not None:
            feature, mask = make_object_tokenizer(dim, w_init=None, name='tokenizer')(feature, one_hot, mask = mask)
        
        # big_pos = jnp.finfo(feature.dtype).max
        big_pos = 999.
        # position = jnp.where(mask, position, big_pos)
        # fts = jnp.squeeze(hk.BatchNorm(True, True, 0.99, name=f'bnfts')(jnp.expand_dims(feature, axis=2), is_training=training, mask=mask), axis=2)
        fts = hk.BatchNorm(True, True, 0.99, name=f'bnfts')(feature, is_training=training, mask=mask)
        
        for idx in range(depth):
            dim_inner = dim * (expansion**idx)
            pts = jnp.where(mask, position, big_pos) if idx == 0 else jnp.where(mask, fts, big_pos)
            fts = make_edge_conv(dim_inner, K, in_depth, name=f'edge_conv{idx}')(pts, fts, training=training, mask=mask)
        
        fts = jnp.where(mask, fts, 0)
        
        return fts, {}
    return particle_net
