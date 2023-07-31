import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["LD_LIBRARY_PATH"] += ':/data/morinaga/cache/pypoetry/virtualenvs/gob-gyPZHm3D-py3.8/lib/python3.8/site-packages/tensorrt/'

import tensorflow.compat.v2 as tf
tf.config.experimental.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')

import numpy as np
import math
import time
from contextlib import contextmanager
from tqdm.auto import tqdm
import jax
from jax import numpy as jnp

import haiku as hk
import optax

class Test1:
    def __init__(self, name):
        self.name = name
        self.times = {}
    
    @contextmanager
    def timer(self, name1):
        start = time.time()
        yield
        end = time.time()
        
        self.times[name1] = end - start
    
    @contextmanager
    def progress_bar(self, epoch, n_epochs, data_length, is_disable_tqdm):
        name = 'Test'
        pbar_args = dict(total = data_length, unit = ' batch', ncols = 250, disable = is_disable_tqdm, bar_format="{desc}{percentage:3.0f}% {n_fmt: >5}/{total_fmt: >5} [{rate_fmt: >16}{postfix}]",)
        pbar_desc = f'{name} [{epoch: >4}/{n_epochs}]'
        
        self.pbar = tqdm(**pbar_args)
        self.pbar.set_description(pbar_desc)
        yield
        
        
    def pbar_update(self):
        self.pbar.set_postfix_str('...')
        self.pbar.update(1)

#from flax.metrics import tensorboard

def test_profile():
    import jax.profiler
    server = jax.profiler.start_server(9999)
    jax.profiler.start_trace("logs/test2")
    
    # Run the operations to be profiled
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5000, 5000))
    y = x @ x
    y.block_until_ready()
    
    jax.profiler.stop_trace()

def test_profile1():
    x = np.random.randn(100)
    y = x.cumsum()
    writer = tensorboard.SummaryWriter('logs/test1')
    for i in range(100):
        writer.scalar("x", x[i], i)
        writer.scalar("y", y[i], i)
    writer.close()

def test_profile2():
    import jax
    from jax import numpy as jnp
    
    @jax.named_call
    def foo(x, y):
        return (x + y) / 2.
    
    @jax.jit
    def bar(a):
        def foo2(x, y):
            return foo(x, y), None
    
        out, _ = jax.lax.scan(foo2, 0., a)
        return out
    
    a = jnp.array([1., 2., 3., 4., 5.])
    
    jax.profiler.start_trace('logs/test6')
    with jax.profiler.StepTraceAnnotation('test', step=0): # JIT warm-up
        out = bar(a)
    with jax.profiler.StepTraceAnnotation('test', step=1):
        out = bar(a)
    out.block_until_ready()
    jax.profiler.stop_trace()

def test_pbar():
    
    
    test = Test1('Test1')
    
    with test.timer('test1'):
        time.sleep(1)
    
    print(test.times)
    with test.progress_bar(1, 1, 10, False):
        for i in range(10):
            time.sleep(1)
            test.pbar_update()

def test_queue():
    import multiprocessing.dummy as mp
    def _generator_process(generator, in_q, out_q):
        for example in generator:
            in_q.get()
            print(f'{example} is putting...')
            out_q.put(example)
    
    def prefetch(n_prefetch = 2):
        def _prefetch(generator):
            in_q, out_q = mp.Queue(), mp.Queue()
            p = mp.Process(target=_generator_process, args=(generator, in_q, out_q))
            for _ in range(n_prefetch):
                in_q.put(None)
            p.start()
            while True:
                yield out_q.get()
                in_q.put(None)
                
                if out_q.qsize() == 0:
                    break
        return _prefetch
    
    
    g = range(20)
    this_prefetch = prefetch(10)
    
    for i in this_prefetch(g):
        print(f'{i} is now here')
        time.sleep(1)

def test_fori_loop():
    rng = jax.random.PRNGKey(0)
    
    images = jax.random.normal(rng, (10, 4, 128, 64, 64))
    
    
    def body(idx):
        image = images[idx]
        jax.device_put_shard()
        out = jnp.mean(out)
        return out
    
    out = jax.lax.fori_loop(0, 9, body, 0)
    print(out)

def test_gob():
    B, N, H, C = 10, 16, 4, 32
    
    n  = jnp.zeros((B, N, H, C))
    en = jnp.zeros((B, H, N, N))
    
    print(f'n   : {n.shape}')
    print(f'en  : {en.shape}')
    
    x = jnp.einsum('b n h c, b h m n -> b m h c', n, en)
    
    print(f'x   : {x.shape}')

def test_gob2():
    B, N, H, C = 10, 16, 4, 32
    q = jnp.zeros((B, N, H, C))
    k = jnp.zeros((B, 1, H, C))
    v = jnp.zeros((B, 1, H, C))
    
    
    qk = jnp.einsum('b n h c, b k h c -> b h k n', q, k)
    print(f'qk   : {qk.shape}')
    
    qkv = jnp.einsum('b h k n, b k h c -> b k n', qk, v)
    print(f'qkv  : {qkv.shape}')
    
    
    edge = jnp.zeros((B, H, N, N))
    print(f'edge : {edge.shape}')
    m_edge = jnp.mean(edge, axis = 1)
    print(f'm_edge : {m_edge.shape}')

def test_gob3():
    B, N, C = 10, 16, 32
    n  = jnp.zeros((B, N, C))
    en = jnp.zeros((B, N, N))
    eg = jnp.zeros((B, 1, N))
    g  = jnp.zeros((B, 1, C))
    
    
    print(f'n   : {n.shape}')
    print(f'en  : {en.shape}')
    print(f'eg  : {eg.shape}')
    print(f'g   : {g.shape}')
    
    
    en1 = jnp.einsum('b n c, b m c -> b n m', n, n)
    print(f'en1 : {en1.shape}')
    
    n1 = jnp.einsum('b n c, b n n -> b n c', n, en1)
    print(f'n1  : {n1.shape}')
    
    
    
    
    eg1 = jnp.einsum('b n c, b k c -> b k n', n, g)
    print(f'eg1 : {eg1.shape}')
    
    
    g1 = jnp.einsum('b n c, b k n -> b k c', n1, eg1)
    print(f'g1  : {g1.shape}')

def test_cbloss():
    n_class = 6
    beta = 0.9999
    gamma = 2.0
    
    labels = jax.random.randint(jax.random.PRNGKey(0), shape=(100,), minval=0, maxval=4, dtype=jnp.int32)
    logits = jax.random.uniform(jax.random.PRNGKey(0), shape=(100,n_class), minval=0, maxval=1.0, dtype=jnp.float32)
    one_hot = jax.nn.one_hot(labels, n_class)
    
    counts = jnp.sum(one_hot, axis = 0)
    weight = (1.0 - beta) / (1.0 - jnp.power(beta, counts))
    print(weight)
    #weight = weight.at[weight == float('inf')].set(0.0)
    weight = jnp.nan_to_num(weight, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(weight)
    weight = weight / jnp.sum(weight, axis = -1)
    
    
    sm = jax.nn.softmax(logits, axis = 1)
    focal = jnp.power(1 - sm, gamma)
    
    print(f'focal  : {focal[:1]}')
    
    
    weight = weight * focal * one_hot
    print(f'weight : {weight.shape}')
    print(f'weight : {weight[:1]}')
    
    
    
    l = weight * jnp.log(sm)
    print(f'l : {l[:1]}')
    
    ce_loss = - jnp.sum(l, axis = -1)
    
    print(f'ce_loss : {ce_loss[:1]}')

def test_tf1():
    table = tf.convert_to_tensor([0,0,1,1])
    
    label1 = tf.convert_to_tensor([0,1,2,3])
    new_label1 = tf.gather(table, label1)
    print(new_label1)
    
    label2 = tf.convert_to_tensor([1,2,0,3,1,0,3,1,1])
    new_label2 = tf.gather(table, label2)
    print(new_label2)
    
    label3 = tf.convert_to_tensor([1,2,3,4,5,6])
    table2 = tf.convert_to_tensor([7,1,1,2,3,4,0])
    table3 = tf.convert_to_tensor([1,1,2,3,4,0])
    new_label3 = tf.gather(table2, label3)
    new_label4 = tf.gather(table2, label3)
    print(new_label3)
    print(new_label4)

def test_pytree():
    
    #from jax.tree_util import tree_flatten, tree_unflatten
    
    params = {
        'g_mlp':{ 'bias' : jnp.zeros((2,2)), 'offset':jnp.zeros((2,2)),
                'g_mlp/linear_1':{'bias':jnp.zeros((2,2)), 'weight':jnp.zeros((2,2))},
                'g_mlp/linear_2':{'bias':jnp.zeros((2,2)), 'weight':jnp.zeros((2,2))},
        }
    }
    
    jax.tree_util.tree_map(lambda x: print(x.shape), params)
    
    _, treedef = jax.tree_util.tree_flatten(params)
    print(treedef)
    
    print(type(params))
    print(type(params['g_mlp']))
    
    flatten = param_flatten(params)
    for k, v in flatten.items():
        print(f'{k} : {v}')

def param_flatten(param, key = '', ret = {}):
    
    if type(param) is not dict:
        ret[key] = param
        return ret
    
    elif type(param) is dict:
        for k in param.keys():
            ret = param_flatten(param[k], f'{key}/{k}', ret)
    
    return ret

def test_GPUtil():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    import GPUtil
    
    for i, res in enumerate(GPUtil.getGPUs()):
        print(f'{i} : {res.load*100}')

def flatten_nested_dict(nested, separator = '/'):
    stack = list(nested.items())
    ans = {}
    while stack:
        key, val = stack.pop()
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                stack.append((f"{key}{separator}{sub_key}", sub_val))
        else:
            ans[key] = val
    return ans

def test_dict():
    d = {'a':1, 'b':{'c':2, 'd':3, 'e':{'f':4, 'e':{'g':5}}, 'h':6}, 'i':7}
    dd = flatten_nested_dict(d, '/')
    for k, v in dd.items():
        print(k, v)

def test_jax1():
    shape = (100, 16, 16, 3)
    a = jax.core.ShapedArray(shape, jnp.float32)
    #a = jnp.ones(shape, jnp.float32)
    print(a.shape)
    #import einops
    
    a = jnp.reshape(a, (100, 16, 4, 12))
    
    
    
    #patch = einops.rearrange(a, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4)
    #print(patch.shape)

def test_patch():
    
    H = 8
    W = 8
    P = 4
    h = H//P
    w = W//P
    print(h,w)
    image = jnp.reshape(jnp.arange(H*W), [H,W,1])
    print(image)
    
    patch = jnp.reshape(image, [h,P,w,P,1])
    patch = jnp.reshape(patch, [h*w,P*P])
    print(patch)
    
    
    import einops
    patch2 = einops.rearrange(image, '(h p1) (w p2) C -> (h w) (p1 p2 C)', p1 = P, p2 = P)
    print(patch2)

def test_patch2():
    
    
    H = 6
    W = 6
    P = 2
    h = H//P
    w = W//P
    C = 2
    print(h,w)
    image = jnp.reshape(jnp.arange(H*W*C), [H,W,C])
    
    print('image : ')
    print(image[:,:,0])
    print(image[:,:,1])
    
    patch = jnp.reshape(image, [h,P,w,P,C])
    # print('patch1 : 1')
    # print(patch)
    patch = jnp.reshape(patch, [h*w,P*P*C])
    print('patch1 :')
    print(patch)
    
    
    import einops
    patch2 = einops.rearrange(image, '(h p1) (w p2) C -> (h w) (p1 p2 C)', p1 = P, p2 = P)
    print('patch2 : ')
    print(patch2)
    
    
    patch3 = jnp.expand_dims(image, 1)
    patch3 = jnp.reshape(patch3, [h,P,W,C])
    patch3 = jnp.expand_dims(patch3, 3)
    patch3 = jnp.reshape(patch3, [h,P,w,P,C])
    patch3 = jnp.swapaxes(patch3, 1, 2)
    patch3 = jnp.reshape(patch3, [h*w, P*P*C])
    print('patch3 :', patch3.shape)
    print(patch3)
    
    patch4 = jnp.expand_dims(image, [1,2])
    patch4 = jnp.reshape(patch4, [h,P,w,P,C])
    patch4 = jnp.swapaxes(patch4, 1, 2)
    patch4 = jnp.reshape(patch4, [h*w, P*P*C])
    print('patch4 :', patch4.shape)
    print(patch4)

def test_imshow():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.zeros([5,5])
    
    eta_idx = [1,2,0,3]
    phi_idx = [0,2,4,1]
    p = [1,2,3,4]
    
    x[eta_idx, phi_idx] = p
    im = ax.imshow(x, origin='lower')
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ax.text(j, i, f'{x[i, j]:.1f}', ha = "center", va = "center", color = "white")
    plt.colorbar(im)
    plt.savefig('figs/test_imshow.png', dpi=300)

def test_tf():
    val = [5,4,3,2,1,0]
    val = [0,1,2,3,4,5]
    table = [1,2,3,0,0,0]
    ret = tf.gather(table, val)
    print(ret)
    
    d = {0:0, 1:1, 2:2, 3:3, 4:0, 23:0, 24:5}
    val = tf.constant([0,1,2,3,4,23,24])
    
    def f(v):
        for k in d.keys():
            if v in [k]:
                return d[k]
        return v

    
    ret = tf.map_fn(f, val)
    print(ret)
    
    d2 = {(0,0):0, (1,1):1, (2,2):2, (3,3):3, (4,4):4, (5,5):5, (1,2):6, (3,4):7}
    val2 = tf.constant([[0,0], [1,1], [2,2], [3,3], [4,4], [1,2], [3,4], [-1,-1]])
    
    def f2(v):
        for key, val in d2.items():
            if np.array_equal(v, key):
                return val
        return 0

    
    ret2 = tf.map_fn(f2, val2)
    print(ret2)

def test_tf_hist():
    import tensorflow as tf
    
    bins = [-1000, 1, 3, 10000]
    vals = [-3, 0, 2, 4, 5, 10, 12]
    
    vals = tf.constant(vals, dtype=tf.float32, name="values")
    bins = tf.constant(bins, dtype=tf.float32, name="bins")
    print(f'bins : {bins}')
    resh_vals = tf.reshape(vals, shape=(1, -1), name="values-reshaped")
    resh_bins = tf.reshape(bins, shape=(-1, 1), name="bins-reshaped")
    print(f'resh_bins : {resh_bins}')
    
    l_bin = tf.less_equal(resh_bins, resh_vals, name="left-edge")
    r_bin = tf.greater(resh_bins, resh_vals, name="right-edge")
    
    resu = tf.logical_and(l_bin[:-1, :], r_bin[1:, :], name="bool-bins")
    counts = tf.reduce_sum(tf.cast(resu, tf.float32), axis=1, name="count-in-bins")
    
    print(counts)

def test_tf_hist2(eps=1e-5):
    import tensorflow as tf
    dR = 1.0
    n_pixel = 10
    bins = tf.linspace(0.0, dR, n_pixel+1)
    print(bins)
    vals = [-1.0, 0.0, 0.11, 0.9, 1.0, 100]
    bins = tf.concat([tf.constant([-eps]), bins, tf.constant([+dR+eps])], axis=0)
    print(bins)
    
    #vals = tf.constant(vals, dtype=tf.float32, name="values")
    bins = tf.constant(bins, dtype=tf.float32, name="bins")
    print(f'bins : {bins}, {bins.shape}')
    resh_vals = tf.reshape(vals, shape=(1, -1), name="values-reshaped")
    resh_bins = tf.reshape(bins, shape=(-1, 1), name="bins-reshaped")
    print(f'resh_bins : {resh_bins}')
    
    l_bin = tf.less_equal(resh_bins, resh_vals, name="left-edge")
    r_bin = tf.greater(resh_bins, resh_vals, name="right-edge")
    print(f'l_bin : {l_bin}')
    print(f'r_bin : {r_bin}')
    
    
    
    resu = tf.logical_and(l_bin[:-1, :], r_bin[1:, :], name="bool-bins")
    print(f'res : {resu}')
    counts = tf.reduce_sum(tf.cast(resu, tf.float32), axis=1, name="count-in-bins")
    print(f'counts : {counts}')
    
    
    index = tf.argmax(resu, axis=0)
    print(f'index : {index}')
    index -= 1
    print(f'index : {index}')
    mask = index >= 0
    print(f'mask : {mask}')

def test_tf_image2():
    n_pixel = 5
    n_ch = 1
    image_shape = [n_ch, n_pixel*n_pixel]
    image = tf.zeros(shape = image_shape, dtype=tf.dtypes.float32)
    
    eta_idx = tf.constant([0,1,2,3,4], dtype=tf.dtypes.int32)
    phi_idx = tf.constant([0,1,2,3,4], dtype=tf.dtypes.int32)
    ch = tf.constant([1,1,2,3,4], dtype=tf.dtypes.float32)
    ch = tf.reshape(ch, (1,-1))
    
    idx = eta_idx + phi_idx*n_pixel
    idx = tf.reshape(idx, (1,-1))
    print(idx)
    print(ch)
    image = tf.tensor_scatter_nd_add(image, idx, ch)
    print(image)

def test_tf_string():
    table = {'h -> ss':1, 'h -> bb':2, 'g -> other':3}
    label = tf.constant(['h -> ss'])
    def cast_str(l):
        key = l.numpy().decode('utf-8')
        return table[key]
    def f(data):
        return tf.py_function(func = cast_str, inp = data, Tout = tf.int32)
    
    ll = f(label)
    print(ll)

def test_tf_tensor1():
    
    t = tf.zeros((4,4,3))
    idx = [(1,1), (2,2), (3,3)]
    v = ([1,1,1],[2,2,2],[3,3,3])
    print(t)
    t2 = tf.tensor_scatter_nd_update(t,idx,v)
    print(t2)
    
    print('---------------------------')
    t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
    t2 = tf.ragged.constant([[6], [7, 8, 9]])
    t  = tf.ragged.stack([t1, t2], axis=0)
    print(t)

def test_tf_image1():
    A = tf.zeros((1,4,4,), dtype=tf.dtypes.float32)
    updates =  tf.constant([1., 2., 3.])
    indices = tf.constant([[0,1,1], [0,2,2], [0,3,3]])
    updated = tf.tensor_scatter_nd_add(A, indices, updates)
    print('A       = ', A)
    print('indices = ', indices)
    print('updates = ', updates)
    
    print('updated = ', updated)
    
    eta = tf.constant([1,2,3])
    phi = tf.constant([1,2,3])
    idx = tf.stack([tf.constant([0]*eta.shape[0]), eta, phi], axis=1)
    print(idx)
    
    image = tf.zeros((4, 32,32))
    print(image.shape)
    image2 = tf.transpose(idx, perm = (1,2,0))
    print(image2.shape)

def test_tf_image3():
    dR = 1.0
    n_pixel = 32
    eps = 1.0e-5
    img_bin = tf.linspace(-dR, +dR, n_pixel + 1)
    img_bin = tf.concat([tf.constant([-dR-eps]), img_bin, tf.constant([dR + eps])], axis = 0)
    img_bin = tf.reshape(img_bin, shape = (-1, 1))
    ch_list = ['track', 'd0']
    
    def binning(val):
        val = tf.reshape(val, shape = (1, -1))
        l_bin = tf.less_equal(img_bin, val)
        r_bin = tf.greater(img_bin, val)
        lr_bin = tf.logical_and(l_bin[:-1,:], r_bin[1:,:])
        index = tf.argmax(lr_bin, axis=0) -1
        mask = index >=0
        return mask, index
    
    def jet_image(data):
        eta = data['eta']
        phi = data['phi']
        eta_mask, eta_idx = binning(eta)
        phi_mask, phi_idx = binning(phi)
        mask = eta_mask & phi_mask
        eta_idx = eta_idx[mask]
        phi_idx = phi_idx[mask]
        
        idx = tf.stack([eta_idx, phi_idx], axis=1)
        image = []
        im_shape = (1, n_pixel, n_pixel)
        im_axis = 0
        for i, key in enumerate(ch_list):
            ch = tf.boolean_mask(data[key], mask, axis=0)
            im = tf.zeros([n_pixel, n_pixel])
            im = tf.tensor_scatter_nd_add(im, idx, ch)
            im = tf.reshape(im, im_shape)
            image.append(im)
        image = tf.concat(image, axis=im_axis)
        return image
    
    data = {}
    n_sub = np.random.randint(4,10, size=(1,))
    
    data['eta'] = tf.constant(np.random.rand(n_sub[0]), dtype=tf.dtypes.float32)
    data['phi'] = tf.constant(np.random.rand(n_sub[0]), dtype=tf.dtypes.float32)
    data['track'] = tf.constant(np.random.rand(n_sub[0]), dtype=tf.dtypes.float32)
    data['d0'] = tf.constant(np.random.rand(n_sub[0]), dtype=tf.dtypes.float32)
    data['calo'] = tf.constant(np.random.rand(n_sub[0]), dtype=tf.dtypes.float32)
    
    image = jet_image(data)
    print(image.shape)

def test_np_vec():
    def f(a,b):
        if a < b:
            return a+b
        return a-b
    vf = np.vectorize(f)
    a = [1,2,3,4]
    b = 2.5
    print(vf(a,b))

    a = [1,2,3,4]
    b = [2,1,4,3]
    print(vf(a,b))

def test_tqdm():
    from tqdm.auto import tqdm
    N = 10
    with tqdm(total=N) as pbar:
        for i in range(N):
            time.sleep(0.5)
            pbar.set_postfix_str('this is inside of loop')
            pbar.update(1)
        pbar.set_postfix_str('this is out of loop')
        pbar.update(0)
        time.sleep(5)
        pbar.set_postfix_str('this is out of loop!!!!!!!!!')
        pbar.update(0)

def test_kwargs():
    def test1(**kwargs):
        print(kwargs)
        if 'image' in kwargs:
            print('!!!!')
    test1()
    test1(image = 'image')

def test_lr():
    import optax
    import jax.numpy as jnp

    @optax.inject_hyperparams
    def optimizer0(learning_rate, eps=1e-8):
        return optax.chain(
            optax.scale_by_rss(initial_accumulator_value=0.0, eps=eps),
            optax.scale(learning_rate),
        )
    @optax.inject_hyperparams
    def optimizer1(learning_rate, eps=1e-8):
        opts = [
            optax.clip_by_global_norm(max_norm = 1.0),
            optax.adam(learning_rate, b1=0.9),
            ]
        return optax.chain(*opts)
    
    #tx = optimizer1(optax.linear_schedule(0.1, 0.0001, 10))
    tx = optimizer1(optax.warmup_cosine_decay_schedule(0.1, 0.2, 2, 10, 0.01))
    opt_state = tx.init({'w': jnp.ones((5, 5)), 'b': jnp.zeros((5))})
    print(opt_state.hyperparams['learning_rate'])
    grads = {'w': jnp.full((5, 5), 0.1), 'b': jnp.full((5), 0.1)}
    
    for i in range(11):
        updates, new_opt_state = tx.update(grads, opt_state)
        print(new_opt_state.hyperparams['learning_rate'])
        opt_state = new_opt_state

def test_einops():
    import einops
    b = 10
    q = 4
    k = 4
    d = 3
    
    din = jnp.zeros((b,q,k,d), dtype = jnp.float32)
    out = einops.rearrange(din, 'b q k d -> b q (k d)')
    print(f'din : {din.shape} {din.dtype}')
    print(f'out : {out.shape} {out.dtype}')

def test_tqdm():
    from time import sleep
    total = 1000
    with tqdm(total=total, bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]") as pbar:
        for i in range(total):
            pbar.update(1)
            sleep(0.1)

def test_rng():
    seed = 1
    rng = jax.random.PRNGKey(seed)
    rng, _ = jax.random.split(rng)
    print('jax')
    for _ in range(10):
        #print(jax.random.bernoulli(rng, 0.5, shape=(2,2)))
        #rng, rng_next = jax.random.split(rng)
        print(jax.random.uniform(rng, shape=(5,)))
    
    import alpa
    alpa.init()
    print('alpa')
    alpa.set_seed(seed)
    alpa.monkey_patch.monkey_patch_random()
    
    for i in range(10):
        print(jax.random.uniform(None, shape=(5,)))
        # print(jax.random.bernoulli(None, 0.5, shape=(5,)))

def test_bfloat():
    a = np.array([math.pi], dtype=np.float16)
    b = tf.constant([math.pi], dtype=tf.bfloat16)
    c = jnp.array([math.pi], dtype=jnp.bfloat16)
    d = b.numpy()
    e = b._numpy()
    def print_ary(x, s):
        print(f'{s} : {x} {x.dtype} {type(x)}')
    print_ary(a, 'a')
    print_ary(b, 'b')
    print_ary(c, 'c')
    print_ary(d, 'd')
    print_ary(e, 'e')

def test_image_grid():
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    
    N = 16
    D = 2/N
    image = np.zeros(shape=(N,N))
    
    C = (1,1)
    def delta(a,b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2
    def inside_of_dR(i,j):
        points = [[D*i,     D*j],
                  [D*(i+1), D*j],
                  [D*(i+1), D*(j+1)],
                  [D*i,     D*(j+1)]]
        r2 = min([delta(p,C) for p in points])
        return r2 <= 1
    
    for i in range(N):
        for j in range(N):
            image[i,j] = inside_of_dR(i,j)
    print(image)
    
def test_padded_batch():
    X = tf.range(10)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset2 = dataset.repeat(3).padded_batch(7, padded_shapes=([8]))
    for item in dataset2:
        print(item)

def test_padding():
    
    # t = tf.constant(list(range(4*2*8)))
    # t = tf.reshape(t, (4,2,8))
    # print(t)
    # #paddings = tf.constant([[0, 0,0], [0, 0, 10]])
    # paddings = tf.constant([[0, 0], [0, 0]])
    # # 'constant_values' is 0.
    # # rank of 't' is 2.
    # tp = tf.pad(t, paddings, "CONSTANT")
    # print(tp)
    
    x = np.stack([np.arange(10*(i+1), 10*(i+1)+10) for i in range(4)])
    print(x)
    print(x.shape)
    x = np.transpose(x,[1,0])
    print(x)
    print(x.shape)
    xp = np.pad(x, ((0,10),(0,0)))
    print(xp)
    print(xp.shape)

import einops
def test_mask():
    rng = jax.random.PRNGKey(0)
    B = 2
    N = 3
    D = 2
    mask = jax.random.randint(rng, shape=(B, N, 1), minval=0, maxval=2)
    print(mask)
    m = einops.repeat(mask==1, 'b n () -> b n d', d = D)
    print(m.shape)
    x = jax.random.uniform(rng, shape=(B, N, D), dtype=jnp.float32)
    xx = jnp.where(m, x, 0)
    print(x)
    print(xx)

def test_tome():
    rng = jax.random.PRNGKey(0)
    B = 2
    N = 20
    D = 4
    R = 0.5
    x = jax.random.uniform(rng, shape=(B, N, D), dtype=jnp.float32)
    x = jnp.ones(shape=(B,N,D), dtype=jnp.float32)
    for i in range(N):
        v = 1.0 + (i+1)*0.1
        x = x.at[0, i, :].multiply(v)
        x = x.at[1, i, :].multiply(-v)
    mask_idx = jax.random.randint(rng, shape=(B, ), minval=3, maxval=N)
    #mask = jax.vmap(lambda a, m: jnp.array([1]*a + [0]*(m-a)), (0, None), 0)(mask_idx, N)
    print(mask_idx)
    print(x)
    def f2(m, n):
        a = jnp.zeros(shape=(N,1), dtype=jnp.int32)
        return a.at[:m].set(1)
    
    mask = jnp.stack([f2(mask_idx[idx], N) for idx in range(B)], axis=0) == 1
    mask_D = einops.repeat(mask, 'b n () -> b n d', d=D)
    xx = x[mask_D]
    print(mask_D.shape)
    print(xx.shape)
    print(xx)
    
    def f3(M,N,rng):
        jnp.shuffle(rng, jnp.arange(M))
    
    # mm = jnp.sum(mask, axis=1)
    # print(mm)
    # mmR = mm*R
    
    # print(mmR)
    
def test_onehot():
    B = 10
    C = 8
    D = 4
    
    rng = jax.random.PRNGKey(0)
    labels = jax.random.randint(rng, shape=(B,), minval=0, maxval=C)
    one_hot = jax.nn.one_hot(labels, C)
    sg = jnp.ones(shape=(B,D), dtype=jnp.float32)
    print(f'one_hot : {one_hot.shape}')
    
    xx = jnp.einsum('BE,BC -> EC', sg, one_hot)
    print(xx.shape)

def test_fcd():
    B = 10
    C = 8
    D = 4
    rng = jax.random.PRNGKey(0)
    x = jax.random.uniform(rng, shape=(B,C,D))
    
    def fcd(x):
        idx = jnp.arange(x.shape[-1])
        xi = x[:, :, idx]
        xu = x[:, :, idx]
        jnp.dot(xi, xu)
        
def test_args():
    
    def f1(x):
        print(f'f1 {x}')
    def f2(x, y, *args):
        print(f'f2 {x} {y}, {args}')
        
    def f3(x, y, *args, t = '', **kwargs):
        print(f'f3 {x} {y}, {args} {t} {kwargs}')
    
    def f(x, *args, t = '', **kwargs):
        print(f'f {x} {args} {t} {kwargs}')
        f1(x)
        f2(x, *args)
        f3(x, *args, t = t, **kwargs)
        
    f('a', 'b', 'c', t = 'd', e = 'e', f = 'f')
    
def test_tome():
    
    B = 4
    N = 32
    D = 8
    r = 2
    rng = jax.random.PRNGKey(0)
    rng, rng2 = jax.random.split(rng)
    x = jax.random.uniform(rng2, shape=(N, D), dtype=jnp.float32)
    rng, rng2 = jax.random.split(rng)
    rand_idx = jnp.argsort(jax.random.uniform(rng2, shape= (N,1), dtype=jnp.float32), axis=0)
    a_idx = rand_idx[:r, :]
    b_idx = rand_idx[r:, :]
    print(rand_idx)
    print(a_idx)
    
    m = jax.random.uniform(rng2, shape=(B, N, D), dtype=jnp.float32)
    a, b = m[..., ::2, :], m[..., 1::2, :]
    score = a @ jnp.transpose(b, (0, 2, 1))
    print(score.shape)

def test_mask():
    a = np.ones([10, 4])
    mask1 = np.squeeze(np.random.uniform(0,1, size=[10,1]) > 0.5)
    mask2 = ~mask1
    x = np.arange(1,5).reshape((1,4))
    x = np.repeat(x, 20, axis=0)
    print(x)
    
    
    print(mask1.shape)
    print(x.shape)
    
    y = np.zeros([10,2])
    len_mask1 = mask1.shape[0]
    len_mask2 = mask2.shape[0]
    y[:len_mask1][mask1, :2] = x[:len_mask1][mask1,:2]
    y[:len_mask2][mask2, :2] = x[:len_mask2][mask2,:2]
    print(y)
    
    # one_hot = np.zeros([10,2])
    # one_hot[mask1, 0] = 1
    # one_hot[mask2, 1] = 1
    # print(one_hot)
    
    mask_test = np.squeeze(np.random.uniform(0,1, size=[10,1]) > 1.5)
    print(mask_test)
    print(mask_test.shape)
    
def test_matmul():
    N = 4
    d = 4
    a = np.arange(N*d).reshape(N,d) + 1
    b = np.arange(N*d).reshape(N,d) + 1
    b = b * 0.1
    c = np.ones(N*d).reshape(N,d) * 0.1
    ab = a * b
    #ac = a @ c
    print(a)
    print(b)
    print(ab)
    #print*(ac)

def test_gather():
    
    x = jnp.array([[0,1,2], [3,4,5]])
    idx = jnp.array([[1], [0]], dtype=jnp.int32)
    #jax.lax.gather(x, idx, )
    
    
    def jax_gather_nd(params, indices):
        tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
        return params[tuple_indices]
    
    print(jax_gather_nd(x,idx))

def test_pn():
    def gather(p,i):
        return p[tuple(jnp.moveaxis(i, -1, 0))]
    
    def gather_nd(p, i):
        return jax.vmap(gather, (0, 0), 0)(p, i)
    
    def gather_2(p, i, K):
        B,N,_ = p.shape
        batch_i = tf.tile(tf.reshape(tf.range(B), (-1, 1, 1, 1)), (1, N, K, 1))
        ii = tf.concat([batch_i, tf.expand_dims(i, axis=3)], axis=3)
        return tf.gather_nd(p, ii)
        
    def gather_3(p, i, K):
        B,N,_ = p.shape
        batch_i = np.tile(np.reshape(np.arange(B), (-1, 1, 1, 1)), (1, N, K, 1))
        ii = np.concatenate([batch_i, np.expand_dims(i, axis=3)], axis=3)
        iii = gather_nd(p, ii) 
        print('gather_3')
        print('batch_i.shape', batch_i.shape)
        print('i', i.shape)
        print('ii', ii)
        return iii
    def distance(A,B):
        rA = np.sum(A*A, axis=2, keepdims=True)
        rB = np.sum(B*B, axis=2, keepdims=True)
        m = np.matmul( A, np.transpose(B, axes=(0,2,1)))
        D = rA - 2.0 * m + np.transpose(rB, axes=(0,2,1))
        return D
    
    
    B = 3
    N = 4
    C = 5
    K = 2
    
    
    
    A = np.random.uniform(size=[B,N,C])
    A[:, 3:, :] = np.finfo(np.float32).max
    print('A')
    print(A.shape)
    print(A)
    # A = np.random.uniform(size=[B,N,C])
    R = distance(A, A)
    print('R')
    print(R.shape)
    print(R[0])
    _, indices = jax.lax.top_k(-R, K+1)
    indices = indices[:, :, 1:]
    print('topk indices')
    print(indices.shape)
    print(indices[0])
    
    one_hot = jax.nn.one_hot(indices, num_classes = N, axis=-1)
    print('one_hot')
    print(one_hot.shape)
    print(one_hot[0])
    
    
    X = np.arange(B*N*C).reshape([B,N,C])
    X = np.random.uniform(size=[B,N,C])
    print('X')
    print(X.shape)
    print(X[0])
    
    x = jnp.einsum('BNC,BMKN -> BMKC', X, one_hot)
    print('x')
    print(x.shape)
    print(x[0])

def test_distance():
    B = 3
    N = 4
    C = 5
    K = 2
    
    position = np.random.uniform(size=[B,N,C])
    def distance(A,B):
        r_A = jnp.sum(A*A, axis=-1, keepdims=True)
        r_B = jnp.sum(B*B, axis=-1, keepdims=True)
        m = jnp.matmul(A, jnp.transpose(B, axes=(0,2,1)))
        m = jnp.einsum('BNC,BMC -> BNM', A, B)
        D = r_A - 2 * m + jnp.transpose(r_B, axes=(0,2,1))
        # D = r_A + jnp.transpose(r_B, axes=(0,2,1))
        print('jax')
        print('A ', A.shape)
        print('B ', B.shape)
        print('rA', r_A.shape)
        print('rB', r_B.shape)
        print('m ', m.shape)
        print('D ', D.shape)
        return D
    
    def distance_tf(A,B):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        print('tf')
        print('A ', A.shape)
        print('B ', B.shape)
        print('rA', r_A.shape)
        print('rB', r_B.shape)
        print('m ', m.shape)
        print('D ', D.shape)
        return D
    
    D = distance(position, position)
    DD = distance_tf(position, position)
    DDD = jnp.sum( (position[:, None, :] - position[:,:,None] )**2, axis=-1)
    
    
    print('D')
    print(D.shape)
    print(D)
    print('DD')
    print(DD.shape)
    print(DD)
    print('DDD')
    print(DDD.shape)
    print(DDD)
    
    X = np.arange(1.,B*N*C+1.).reshape([B,N,C])
    X[:,3:,:] = np.finfo(np.float32).max
    
    m = jnp.einsum('BNC,BMC -> BNM', X, X)
    print('X')
    print(X)
    print('m')
    print(m)


def test_array_apply():
    B = 2
    N = 4
    C = 3
    x = jnp.ones([B,N,C])
    
    
    def logPt(v):
        return jnp.where(v>0, jnp.log(v), 0)
    
    
    # xx = x.at[:,:,0].apply(logPt)
    x = x.at[:,3:,0].set(0)
    xx = x.at[:,:,0].set(logPt(x[:,:,0]))
    print('x')
    print(x)
    print('xx')
    print(xx)

def test_split():
    x = jnp.ones([5,4, 20])
    xs = jnp.split(x, [5,10], axis=-1)
    print(len(xs))
    for xx in xs:
        print(xx.shape)
        
    
    # m = jnp.reshape(x, [-1, 1])
    # print(m.shape)
    
    # m = jnp.tile(m, [1,2])
    # print(m.shape)
    
    # x1 = jnp.reshape(jnp.arange(6), [1,2,3])
    # x2 = jnp.reshape(jnp.arange(6), [1,2,3])
    # x = jnp.stack([x1,x2], axis=-1)
    # print(x.shape)
    
    # x = jnp.reshape(x, [1,2,-1])
    # print(x.shape)
    # print(x)
    
    # a = jnp.arange(6)
    # b = a[None, :, None]
    # print(b.shape)
    
    # c = jnp.tile(b, [5,1,1])
    # print(c)
    # print(c.shape)
    
    N = 36
    chunk = 6
    a = jnp.linspace(0, N, chunk+1, dtype=jnp.int64)[1:-1]
    print(a)
    b = jnp.ones([3,4,N])
    print(b.shape)
    cs = jnp.split(b, a, axis=-1)
    print(len(cs))
    for c in cs:
        print(c.shape)
        
    
    def ret(A,B):
        return A,B
    
    p = [None]*4
    
    for i in range(4):
        a, p[i] = ret(i+1, i*2+1)
    print('p')
    print(p)

def check_data(): 
    n_data = {
    'total'  : 49892040,
    'H -> ss':  1915408,
    'H -> cc':  1960586,
    'H -> bb':  1862663,
    'Z -> ss':  1937696,
    'Z -> cc':  1982207,
    'Z -> bb':  1871392,
    'u -> xx':  6315093,
    'd -> xx':  5809299,
    's -> xx':  4112375,
    'c -> xx':  4638861,
    'b -> xx':  7692253,
    'g -> xx':  7029733,
    'g -> uu':   719416,
    'g -> dd':   712648,
    'g -> ss':   634734,
    'g -> cc':   441340,
    'g -> bb':   256336,
    }
    
    
    n_data2 = {
        'uu/dd': [719416, 712648],
        'ss': [1915408, 1937696, 634734], 
        'cc': [1960586, 1982207, 441340],
        'bb': [1862663, 1871392, 256336],
        'u/d': [5809299, 6315093],
        's': [4112375],
        'c': [4638861],
        'b': [7692253],
        'g': [7029733],
    }
    total = 0
    for key, val in n_data2.items():
        s = sum(val)
        #print(key, s, s/n_data['total']*100)
        v = s/n_data['total']*100
        print(key, f'{v:.2f}%')
        total += s
    print(total/n_data['total'])
    
    

if __name__ == '__main__':
    #test_profile2()
    #test_queue()
    #test_fori_loop()
    
    #test_gob2()
    #test_cbloss()
    #test_tf()
    #test_tf_hist2()
    #test_tf_string()
    #test_tf_tensor1()
    #test_tf_image1()
    #test_tf_image3()
    #test_np_vec()
    #test_tqdm()
    #test_kwargs()
    #test_lr()
    #test_einops()
    #test_pytree()
    #test_GPUtil()
    #test_dict()
    #test_jax1()
    #test_patch()
    #test_patch2()
    #test_imshow()
    # test_rng()
    #test_bfloat()
    #test_image_grid()
    #test_padded_batch()
    #test_padding()
    #test_mask()
    #test_tome()
    #test_onehot()
    #test_args()
    # test_tome()
    #test_mask()
    # test_matmul()
    # test_gather()
    # test_pn()
    # test_distance()
    #test_array_apply()
    #test_split()
    check_data()
    
    
    
    
    
    
    
    
    
    