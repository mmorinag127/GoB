import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import tensorflow.compat.v2 as tf


import numpy as np
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
        ret[key] = param.shape
        return ret
    
    elif type(param) is dict:
        for k in param.keys():
            ret = param_flatten(param[k], f'{key}/{k}', ret)
    
    return ret
    
def test_GPUtil():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

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
    


    
if __name__ == '__main__':
    #test_profile2()
    #test_queue()
    #test_fori_loop()
    
    #test_gob2()
    #test_cbloss()
    #test_tf1()
    #test_pytree()
    #test_GPUtil()
    test_dict()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    