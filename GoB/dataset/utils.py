


import collections
import itertools
import multiprocessing.dummy as mp

import jax
from jax.interpreters import xla
import jax.lib.xla_bridge as xb

# from flax.jax_utils
def _pmap_device_order():
    # match the default device assignments used in pmap:
    # for single-host, that's the XLA default device assignment
    # for multi-host, it's the order of jax.local_devices()
    if jax.process_count() == 1:
        return [d for d in xb.get_backend().get_default_device_assignment(jax.device_count()) if d.process_index == jax.process_index()]
    else:
        return jax.local_devices()

# from flax.jax_utils
def prefetch_to_device(iterator, size, devices=None):
    """Shard and prefetch batches on device.
    
    This utility takes an iterator and returns a new iterator which fills an on
    device prefetch buffer. Eager prefetching can improve the performance of
    training loops significantly by overlapping compute and data transfer.
    
    This utility is mostly useful for GPUs, for TPUs and CPUs it should not be
    necessary -- the TPU & CPU memory allocators (normally) don't pick a memory
    location that isn't free yet so they don't block. Instead those allocators OOM.
    
    Args:
        iterator: an iterator that yields a pytree of ndarrays where the first
            dimension is sharded across devices.
        size: the size of the prefetch buffer.
            If you're training on GPUs, 2 is generally the best choice because this
            guarantees that you can overlap a training step on GPU with a data
            prefetch step on CPU.
        
        devices: the list of devices to which the arrays should be prefetched.  
            Defaults to the order of devices expected by `jax.pmap`.
    
    Yields:
        The original items from the iterator where each ndarray is now a sharded to
        the specified devices.
    """
    queue = collections.deque()
    devices = devices or _pmap_device_order()
    
    def _prefetch(xs):
        if hasattr(jax, "device_put_sharded"):  # jax>=0.2.0
            return jax.device_put_sharded(list(xs), devices)
        else:
            aval = jax.xla.abstractify(xs)
            assert xs.shape[0] == len(devices), (
                "The first dimension of the iterator's ndarrays is not "
                "equal to the number of devices.")
            buffers = [xla.device_put(x, devices[i]) for i, x in enumerate(xs)]
            return jax.pxla.ShardedDeviceArray(aval, buffers)
    
    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_map(_prefetch, data))
    
    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)

def _prefetch_to_device(size, devices = None):
    
    devices = devices or _pmap_device_order()
    
    def _device_put_shared(xs):
        return jax.device_put_sharded(list(xs), devices)
    
    def _gen_process(generator, i_q, o_q):
        for data in itertools.islice(generator, size):
            i_q.get()
            o_q.put( jax.tree_map(_device_put_shared, data) )
    
    def prefetch(iterator):
        in_q, out_q = mp.Queue(), mp.Queue()
        p = mp.Process(target=_gen_process, args=(iterator, in_q, out_q))
        for _ in range(size):
            in_q.put(None)
        
        p.start()
        while True:
            yield out_q.get()
            in_q.put(None)
            #if out_q.qsize() == 0:
            #    break
    return prefetch
    
def _generator_process(generator, in_q, out_q):
    for example in generator:
        in_q.get()
        out_q.put(example)

def Prefetch(n_prefetch=2):  # pylint: disable=invalid-name
    """Pre-fetches a number of examples from generator in a separate process."""
    def prefetch(generator):
        in_q, out_q = mp.Queue(), mp.Queue()
        p = mp.Process(target=_generator_process, args=(generator, in_q, out_q))
        for _ in range(n_prefetch):
            in_q.put(None)
        p.start()
        while True:
            yield out_q.get()
            in_q.put(None)
    return prefetch

def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()
    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access
        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])
        
    return jax.tree_map(_prepare, xs)






















