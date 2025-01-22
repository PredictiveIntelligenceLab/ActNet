import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, grad


from functools import partial
import torch.utils.data as data

  
# Dataset loader
class BatchedDataset(data.Dataset):
  ''' A data loader for creating mini-batches.

  Attributes:
    raw_data: full dataset to be used. This should be a tuple of lenght 3,
      formated as (inputs, targets, weights).
    key: a PRNG key used as the random key.
    batch_size: the size of each mini-batch. If None, uses full batch
      (default: None).
  '''

  def __init__(self, raw_data, key, batch_size=None):
    super().__init__()
    self.inputs = raw_data[0]
    self.targets = raw_data[1]
    self.weights = raw_data[2]
    self.size = len(self.weights)
    self.key = key
    if batch_size is None: # Will use full batch
      self.batch_size = self.size
    else:
      self.batch_size = batch_size
    
  def __len__(self):
    return self.size
  
  def __getitem__(self, idx):
    self.key, subkey = random.split(self.key)
    batch_inputs, batch_targets, batched_weights = self.__select_batch(subkey)
    return batch_inputs, batch_targets, batched_weights

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, key):
    idx = random.choice(key, self.size, (self.batch_size,), replace=False)
    batch_inputs = self.inputs[idx]
    batch_targets = self.targets[idx]
    batched_weights = self.weights[idx]
    return batch_inputs, batch_targets, batched_weights


class SquareDataset(data.Dataset):
  ''' A data loader for creating mini-batches of uniformly samples points on the
   inside and on the boundary of a [-1,1]^2 square. Generates a pair of vectors
   (interior_batch, border_batch) with iid points on the interior and border of
   squre, respectively.

  Attributes:
    key: a PRNG key used as the random key.
    batch_size: the size of each mini-batch. Should be a tuple of lenght 2 in
      the format (inside_batch_size, border_batch_size).
  '''
  def __init__(self, key, batch_size=(10_000, 800)):
    super().__init__()
    self.size = batch_size[0]
    self.key = key
    self.batch_size = batch_size
    
  def __len__(self):
    return self.size
  
  def __getitem__(self, idx):
    self.key, subkey1, subkey2 = random.split(self.key, 3)
    interior_batch, border_batch = self.__select_batch(subkey1, subkey2)
    return interior_batch, border_batch

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, subkey1, subkey2):
    interior_batch = random.uniform(subkey1, shape=(self.batch_size[0], 2),
                                    minval=-1, maxval=1)
    border_batch = random.uniform(subkey2, shape=(self.batch_size[1],1),
                                  minval=-1, maxval=1)
    aux = jnp.split(border_batch, 4)
    border_batch = jnp.concatenate([
      jnp.hstack([-jnp.ones_like(aux[0]), aux[0]]),
      jnp.hstack([jnp.ones_like(aux[1]), aux[1]]),
      jnp.hstack([aux[2], -jnp.ones_like(aux[2])]),
      jnp.hstack([aux[3], jnp.ones_like(aux[3])]),
      ], axis=0)
    return interior_batch, border_batch

# alias for SquareDataset
Poisson2DDataset = SquareDataset