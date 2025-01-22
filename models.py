import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, grad, value_and_grad

import numpy as onp
import optax
from optax._src import linear_algebra
import jaxopt

from functools import partial
import itertools
from tqdm import trange
import matplotlib.pyplot as plt



# not used for the ActNet paper, but maybe useful for the future
class RegressionModel:
    """ A model for training/evaluating a neural network using regression.

    Attributes:
        arch (nn.Module): a Flax module of the desired architecute.
        batch: an initial batch used for initializing parameters and computing
            normalization factors.
        optimizer: the optimizer to be used when running gradient descent.
        normalize_inputs: whether to normalize inputs before passing them to the
            architecture (default: True).
        normalize_outputs: whether to normalize outputs of the architecture
            (default: True).
        key: a PRNG key used as the random key for initialization.
        steps_per_check (int): how many training steps to use between logging
            and displaying losses (default: 100).
    """
    def __init__(self, arch, batch, optimizer=None, normalize_inputs=True,
                 normalize_outputs=True, key=random.PRNGKey(43),
                 steps_per_check=100) -> None:
        # Define model
        self.arch = arch
        self.key = key
        self.steps_per_check = steps_per_check

        # Initialize parameters
        inputs, outputs, _ = batch
        self.params = self.arch.init(self.key, inputs)

        # Tabulate function for checking network architecture
        self.tabulate = lambda : \
            self.arch.tabulate(self.key, inputs, console_kwargs={'width':110})
        
        # Vectorized functions
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.normalize_data = (normalize_inputs or normalize_outputs)
        if self.normalize_data:
           mu_x = inputs.mean(0, keepdims=True)
           sig_x = inputs.std(0, keepdims=True)
           mu_y = outputs.mean(0, keepdims=True)
           sig_y = outputs.std(0, keepdims=True)
           self.norm_stats = ((mu_x, sig_x), (mu_y, sig_y))
           if self.normalize_inputs:
               if self.normalize_outputs:
                   self.apply = lambda params, x : \
                    mu_y + sig_y*self.arch.apply(params,
                                                 (x-mu_x)/(sig_x + 0.01))
               else:
                   self.apply = lambda params, x : \
                    self.arch.apply(params, (x-mu_x)/(sig_x + 0.01))
           else:
               self.apply = lambda params, x : \
                mu_y + sig_y*self.arch.apply(params, x)
                      
        else:
            self.norm_stats = None
            self.apply = self.arch.apply
        # jits apply function for numerical consistency (sometimes jitted 
        # version behaves slightly differently than non-jitted one)
        self.apply = jit(self.apply)

        # Optimizer
        if optimizer is None:
            lr = optax.exponential_decay(1e-3, transition_steps=1000,
                                         decay_rate=0.9, end_value=1e-5)
            self.optimizer = optax.adam(learning_rate=lr)
        else:
            self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.params)

        # Optimizer LBFGS
        self.optimizer_lbfgs = jaxopt.LBFGS(self.loss)
        self.opt_state_lbfgs = self.optimizer_lbfgs.init_state(self.params,
                                                               batch)
        self.optimizer_update_lbfgs = jit(self.optimizer_lbfgs.update)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.grad_norm_log = []

    def recon_loss(self, params, u, s, w):
        outputs = self.apply(params, u) # shape (batch_dim, out_dim)
        loss = jnp.mean(w*(s-outputs)**2, axis=(-1)) # shape (batch_dim,)
        return loss
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        inputs, targets, weights = batch
        u = inputs
        s = targets
        w = weights
        return self.recon_loss(params, u, s, w).mean() # scalar
    

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, batch):
        grads = grad(self.loss)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grads

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 10000):
        """ Trains the neural network for nIter steps using data loader.

        Args:
            dataset (BatchedDataset): data loader for training.
            nIter (int): number of training iterations.
        """
        data = iter(dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            batch = next(data)
            self.params, self.opt_state, grads = self.step(self.params,
                                                           self.opt_state,
                                                           batch)
            # Logger
            if it % self.steps_per_check == 0:
                l = self.loss(self.params, batch)
                g_norm = linear_algebra.global_norm(grads).squeeze()
                self.loss_log.append(l)
                self.grad_norm_log.append(g_norm)
                pbar.set_postfix({
                    'loss': l,
                    'grad_norm': jnp.mean(jnp.array(g_norm))
                     })

    # Define a compiled update step    
    @partial(jit, static_argnums=(0,))
    def step_lbfgs(self, params, opt_state, batch):
        new_params, opt_state = self.optimizer_update_lbfgs(params,
                                                            opt_state,
                                                            batch)
        return new_params, opt_state

    # Optimize parameters in a loop
    def train_lbfgs(self, dataset, nIter = 10000):
        """ Trains the neural network using LBFGS optimizer for nIter steps
        using data loader.

        Args:
            dataset (BatchedDataset): data loader for training.
            nIter (int): number of training iterations.
        """
        data = iter(dataset)
        pbar = trange(nIter)
        batch = next(data)
        self.opt_state_lbfgs = self.optimizer_lbfgs.init_state(self.params,
                                                               batch)
        # Main training loop
        for it in pbar:
            batch = next(data)
            # Logger
            if it % self.steps_per_check == 0:
                l = self.loss(self.params, batch)
                self.loss_log.append(l)
                grads = grad(self.loss)(self.params, batch)
                g_norm = linear_algebra.global_norm(grads).squeeze()
                self.grad_norm_log.append(g_norm)
                pbar.set_postfix({
                    'loss': l,
                    'grad_norm': jnp.mean(jnp.array(g_norm))
                    })
            # optimization step
            self.params, self.opt_state_lbfgs = self.step_lbfgs(self.params,
                                                                self.opt_state_lbfgs,
                                                                batch)
    
    def plot_logs(self, window=None) -> None:
        """ Plots logs of training losses and gradient norms through training.

        Args:
            window: desired window for computing moving averages (default: None)
        """
        plot_logs(self.loss_log, self.grad_norm_log, window=window,
                  steps_per_check=self.steps_per_check)

    def batched_apply(self, x, batch_size=2_048):
       '''Performs forward pass using smaller batches, then concatenates them
       together before returning predictions. Useful for avoiding OoM issues 
       when input is large.

       Args:
          x: input to the model
          batch_size: maximum batch size for computation.

        Returns:
          predictions of the model on input x
       '''
       num_batches = int(jnp.ceil(len(x) / batch_size))
       x_batches = jnp.split(x,
                             batch_size*(1+jnp.arange(num_batches-1)),
                             axis=0)
       pred_fn = lambda ins : self.apply(self.params, ins)
       y_pred = jnp.concatenate([pred_fn(ins) for ins in x_batches], axis=0)
       return y_pred
    
    def get_rmse(self, batch, batch_size=2_048):
       # Create predictions
        u, s_true, _ = batch
        if batch_size is None: # single forward pass
          s_pred = self.apply(self.params, u)
        else: # breaks prediction into smaller forward passes
          s_pred = self.batched_apply(u, batch_size=batch_size)
        error = s_pred - s_true
        rmse = jnp.sqrt(jnp.mean(error**2))
        return rmse

    def plot_predictions(self, batch, return_pred=False, batch_size=2_048):
        """Computes and plots model predictions for a given batch of data.

        Args:
            batch: data for creating/plotting results.
            return_pred: whether to return predictions after plotting 
                (default: False).
            batch_size: batch size for computations (to avoid OoM issues in the
                case of large datasets). (default: 2048)
        """
        # Create predictions
        u, s_true, _ = batch
        if batch_size is None: # single forward pass
          s_pred = self.apply(self.params, u)
        else: # breaks prediction into smaller forward passes
          s_pred = self.batched_apply(u, batch_size=batch_size)

        error = s_pred - s_true
        rel_l2_error = jnp.sqrt(jnp.sum(error**2)/jnp.sum(s_true**2))
        print('Relative L2 error: {:.2e}'.format(rel_l2_error))
        print('RMSE: {:.2e}'.format(jnp.sqrt(jnp.mean(error**2))))

        if u.shape[-1]== 1: # domain is 1D
            plt.figure(figsize=(15, 4))

            # Ploting examples of reconstructions
            plt.subplot(131)
            plt.plot(u, s_true, lw=1)
            plt.plot(u, s_pred, '--', lw=1)
            plt.xlabel('$y$')
            plt.ylabel('$s$')
            plt.title('Prediction Vs Truth (predictions are dashed)')

            # Ploting error
            plt.subplot(132)
            plt.plot(u, s_pred-s_true, lw=1)
            plt.xlabel('$y$')
            plt.ylabel('$s$')
            plt.title('Error')

            # plotting histogram of errors
            plt.subplot(133)
            plt.hist(error.flatten(), bins=50)
            plt.title('Histogram of errors')
            
            plt.show()
        elif u.shape[-1] == 2: # domain is 2D
            plt.figure(figsize=(15, 4))

            # Ploting examples of reconstructions
            plt.subplot(131)
            plt.scatter(u[:,0], u[:,1], c=s_pred)
            plt.colorbar()
            plt.xlabel('$y$')
            plt.ylabel('$s$')
            plt.title('Prediction')

            # Ploting true solution
            plt.subplot(132)
            plt.scatter(u[:,0], u[:,1], c=s_true)
            plt.colorbar()
            plt.xlabel('$y$')
            plt.ylabel('$s$')
            plt.title('True')

            # Ploting errors
            plt.subplot(133)
            plt.scatter(u[:,0], u[:,1], c=s_pred-s_true)
            plt.colorbar()
            plt.xlabel('$y$')
            plt.ylabel('$s$')
            plt.title('Error')

            plt.show()
        else: # domain is higher than 2D. Plot histogram of errors instead
           # plotting histogram of errors
            plt.hist(error.flatten(), bins=50)
            plt.title('Histogram of errors')
            plt.show()

        if return_pred:
           return s_pred

# alias for RegressionModel
SupervisedModel = RegressionModel



# Functions to help plotting
def plot_logs(loss_log, grad_norm_log, window=None, steps_per_check=100):
  """ Plots logs of training losses and gradient norms through training.

  Args:
    loss_log: sequence of training losses.
    grad_norm_log: sequence of parameter gradient norms.
    window: desired window for computing moving averages (default: None).
    steps_per_check: how many training steps were taken between each log.
  """
  plt.figure(figsize=(12, 4))

  # Plotting losses
  plt.subplot(121)
  if window is None:
      plt.plot(steps_per_check*jnp.arange(len(loss_log)), loss_log)
  else:
      assert type(window) is int , f'window must be an interger or None, not {type(window)}'
      plt.plot(steps_per_check*jnp.arange(len(loss_log) - window),
               [onp.mean(loss_log[i:i+window]) for i in range(len(loss_log) - window)])
  plt.yscale('log')
  plt.title('Loss through iterations')

  # Plotting gradient norms
  plt.subplot(122)
  if window is None:
      plt.plot(steps_per_check*jnp.arange(len(grad_norm_log)), grad_norm_log)
  else:
      assert type(window) is int , f'window must be an interger or None, not {type(window)}'
      plt.plot(steps_per_check*jnp.arange(len(grad_norm_log) - window),
               [onp.mean(grad_norm_log[i:i+window]) for i in range(len(grad_norm_log) - window)])
  plt.yscale('log')
  plt.title('Global gradient norm through iterations')
  plt.show()