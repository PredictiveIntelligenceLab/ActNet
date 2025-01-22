import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, value_and_grad

import optax
from optax._src import linear_algebra

from functools import partial
import itertools
from tqdm import trange
import torch.utils.data as data

import matplotlib.pyplot as plt

import sys
sys.path.append('..') # makes modules in parent repository available to import
from models import plot_logs


class SquareDataset(data.Dataset):
  ''' A data loader for creating mini-batches of uniformly samples points on the
   inside of a square/rectangle. Generates iid points on the interior of the 
   square/rectangle. These points are returned ordered by time (earliest first).

  Attributes:
    key: a PRNG key used as the random key.
    minvals: pair indicating minimum values for each dimension.
    minvals: pair indicating maximum values for each dimension.
    batch_size: the size of each mini-batch.
  '''
  def __init__(self, key, minvals=(-1,-1), maxvals=(1,1), batch_size=10_000):
    super().__init__()
    self.minvals = jnp.array(minvals)
    self.maxvals = jnp.array(maxvals)
    self.size = batch_size
    self.key = key
    self.batch_size = batch_size
    
  def __len__(self):
    return self.size
  
  def __getitem__(self, idx):
    self.key, subkey = random.split(self.key)
    interior_batch = self.__select_batch(subkey)
    return interior_batch

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, subkey):
    interior_batch = random.uniform(subkey, shape=(self.batch_size, 2),
                                    minval=self.minvals, maxval=self.maxvals)
    # time is last coordinate
    idx_sort = jnp.argsort(interior_batch[:,-1])
    # returns batch with times ordered in increasing time
    return interior_batch[idx_sort]
  

def get_ac_residual_fn(f, D=1e-4):
    '''
    Computes the Allen-Cahn (AC) residual of a given funciton in a
    computationally efficient way by avoiding unecessary repetition of the
    computational graph.

    Args:
        f (Callable): a function with signature (params, x, t)->scalar
        D (float): the D constant parameter for the AC equation.
    Returns:
        (Callable): a function with signature (params, x, t)->scalar that is the
                    AC residual of input function
    '''
    _single_f = lambda params, x, t : f(params, x[None,:], t[None,:]).squeeze()
    def _scalar_grads(params, x, t):
        u, (ux, ut) = value_and_grad(_single_f, argnums=(1,2))(params, x, t)
        return ux.squeeze(), (u, ut.squeeze())
    def _ac_aux(params, x, t):
        (u_x, (u, u_t)), u_xx =  value_and_grad(_scalar_grads,
                                                argnums=1,
                                                has_aux=True)(params, x, t)
        return u, u_t, u_xx.squeeze()
    def _res_fn(params, xs, ts):
        u, ut, uxx = vmap(_ac_aux, in_axes=(None, 0, 0))(params, xs, ts)
        return ut - D*uxx + 5*(u**3 - u)
    return _res_fn



class AllenCahnModel:
    """ A model for training/evaluating a neural network using physics-informed
    causal training on the Allen-Cahn problem.

    Attributes:
        arch (nn.Module): a Flax module of the desired architecute.
        batch: an initial batch used for initializing parameters and computing
            normalization factors.
        true_fun: the true ground_truth function, if known (default: None).
        D: the D constant parameter for the AC equation.
        optimizer: the optimizer to be used when running gradient descent.
        normalize_inputs: whether to normalize inputs before passing them to the
            architecture (default: True).
        key: a PRNG key used as the random key for initialization.
        exact_bd_condition: whether to exactly enforce boundary condition on 
            [-1,1]^2 (see  https://arxiv.org/abs/2104.08426). This implicitly
            makes it so that border losses are not computed (defaut: True).
        bdr_enforcer_order: order of the polynomial used for enforcing border 
            exactly. Should be an even integer (default: 2).
        steps_per_check (int): how many training steps to use between logging
            and displaying losses (default: 50).
    """
    def __init__(self, arch, batch, true_sol=None, D=1e-4,
                 optimizer=None, normalize_inputs=True, key=random.PRNGKey(43),
                 exact_bd_condition=True, bdr_enforcer_order=2,
                 steps_per_check=50) -> None:
        # Define model
        self.arch = arch
        self.key = key
        self.steps_per_check = steps_per_check
        self.true_sol = true_sol
        self.D = D

        # Initialize parameters
        interior_batch = batch
        self.params = self.arch.init(self.key, interior_batch)

        # Tabulate function for checking network architecture
        self.tabulate = lambda : \
            self.arch.tabulate(self.key,
                               interior_batch,
                               console_kwargs={'width':110})
        
        # Vectorized functions
        self.normalize_inputs = normalize_inputs
        self.exact_bd_condition = exact_bd_condition
        self.bdr_enforcer_order = bdr_enforcer_order # should be an even number
        if normalize_inputs:
            mu_x = jnp.hstack(interior_batch).mean(0, keepdims=True)
            sig_x = jnp.hstack(interior_batch).std(0, keepdims=True)
            self.norm_stats = (mu_x, sig_x)
            _apply = lambda params, x, y : \
                self.arch.apply(params, (jnp.hstack([x, y])-mu_x)/sig_x)
            if self.exact_bd_condition:
               _apply = lambda params, x, t : \
                (1-t)*(x**2 * jnp.cos(jnp.pi*x)) \
                    + t*((1-x**self.bdr_enforcer_order)*self.arch.apply(params,
                                                                        (jnp.hstack([x, t])-mu_x)/sig_x) - 1)
            else:
               _apply = lambda params, x, y : \
                self.arch.apply(params, (jnp.hstack([x, y])-mu_x)/sig_x)
        else:
            self.norm_stats = None
            if self.exact_bd_condition:
               _apply = lambda params, x, t : \
                (1-t)*(x**2 * jnp.cos(jnp.pi*x)) \
                    + t*((1-x**self.bdr_enforcer_order)*self.arch.apply(params,
                                                                        jnp.hstack([x, t])) - 1)
            else:
               _apply = lambda params, x, y : \
                self.arch.apply(params, jnp.hstack([x, y]))
        # jits apply function for numerical consistency (sometimes jitted 
        # version behaves slightly differently than non-jitted one)
        self.apply = jit(_apply)

        # Vectorized derivatives.
        # functions prefixed by '_single' take in a vector of shape (1,) and
        # output a scalar of shape (,)
        _single_f = lambda params, x, y : \
            self.apply(params, x[None,:], y[None,:]).squeeze()
        # x derivatives
        _single_f_x = lambda params, x, y : \
            grad(_single_f, argnums=1)(params, x, y).squeeze() # scalar
        self.f_x = vmap(_single_f_x, in_axes=(None, 0, 0))
        _single_f_xx = lambda params, x, y : \
            grad(_single_f_x, argnums=1)(params, x, y).squeeze() # scalar
        self.f_xx = vmap(_single_f_xx, in_axes=(None, 0, 0))
        # y derivatives
        _single_f_y = lambda params, x, y : \
            grad(_single_f, argnums=2)(params, x, y).squeeze() # scalar
        self.f_y = vmap(_single_f_y, in_axes=(None, 0, 0))
        _single_f_yy = lambda params, x, y : \
            grad(_single_f_y, argnums=2)(params, x, y).squeeze() # scalar
        self.f_yy = vmap(_single_f_yy, in_axes=(None, 0, 0))
        # laplacian
        self.ac_residual = get_ac_residual_fn(self.apply, D=self.D)

        # Optimizer
        if optimizer is None: # use a standard optimizer
            lr = optax.exponential_decay(1e-3, transition_steps=1000,
                                         decay_rate=0.8, end_value=1e-7)
            self.optimizer = optax.chain(
               optax.adaptive_grad_clip(1e-2),
               optax.adam(learning_rate=lr),
               )
        else:
            self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.grad_norm_log = []
        self.rel_l2_log = []
    
    def residual_loss(self, params, x, t, causal_eps):
        res = self.ac_residual(params, x, t)[:,None] # shape (batch_dim,1)
        goal = jnp.zeros_like(res)
        res = jnp.mean((res-goal)**2, axis=-1) # shape (batch_dim,)
        if causal_eps is None: # no causal learning
            return res
        else:
            # compute causal weights
            ws = jax.lax.stop_gradient(jnp.exp(-causal_eps*(jnp.cumsum(res) - res))) # shape (num_ts,)
            # make it so that mean value of weights is 1 to maintain loss in the
            # same order of magnitude
            ws = ws/(ws.mean()+1e-3)
            assert ws.shape == res.shape, f"ws is shape {ws.shape} but res is shape {res.shape}"
            return ws*res



    def pinn_loss(self, params, interior_batch, causal_eps):
        r_loss = self.residual_loss(params,
                                    interior_batch[:,0][:,None],
                                    interior_batch[:,1][:,None],
                                    causal_eps)
        if self.exact_bd_condition:
            # no need to consider border loss, since it will be 0 when bdry 
            # condition is exactly enforced
            return r_loss.mean()
        else:
            raise NotImplementedError
            # consider both residual loss initial condition loss and boundary condition loss
            #b_loss = self.border_loss(params, border_batch[:,0][:,None], border_batch[:,1][:,None])
            #return self.pinn_weights[0]*r_loss.mean() + self.pinn_weights[1]*b_loss.mean()    

    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch, causal_eps):
        interior_batch = batch
        return self.pinn_loss(params, interior_batch, causal_eps).mean() # scalar
    

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, batch, causal_eps):
        grads = grad(self.loss)(params, batch, causal_eps)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grads

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 10_000, causal_eps=None):
        """ Trains the neural network for nIter steps using data loader.

        Args:
            dataset (SquareDataset): data loader for training.
            nIter (int): number of training iterations.
            causal_eps (None | float): epsilon for computing causal loss. If
                None, does not use causal learning (default: None).
        """
        data = iter(dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            batch = next(data)
            self.params, self.opt_state, grads = self.step(self.params,
                                                           self.opt_state,
                                                           batch,
                                                           causal_eps)
            # Logger
            if it % self.steps_per_check == 0:
                l = self.loss(self.params, batch, causal_eps)
                g_norm = linear_algebra.global_norm(grads).squeeze()
                self.loss_log.append(l)
                self.grad_norm_log.append(g_norm)
                if self.true_sol is not None:
                    pred = self.apply(self.params,
                                      self.true_sol[0][0],
                                      self.true_sol[0][1])
                    true = self.true_sol[1]
                    rel_l2_error = jnp.sqrt(((pred-true)**2).mean() \
                                            / ((true)**2).mean())
                    self.rel_l2_log.append(rel_l2_error)
                    pbar.set_postfix_str(f"loss:{l : .3e}, rel_l2:{rel_l2_error : .2e}, 'grad_norm':{jnp.mean(jnp.array(g_norm)) : .2e}")
                else:
                    pbar.set_postfix({
                       'loss': l,
                       'grad_norm': jnp.mean(jnp.array(g_norm)),
                       })

    def plot_logs(self, window=None) -> None:
        """ Plots logs of training losses and gradient norms through training.

        Args:
            window: desired window for computing moving averages (default: None).
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
       pred_fn = jit(lambda ins : \
                     self.apply(self.params,
                                ins[:,0][:,None],
                                ins[:,1][:,None]))
       y_pred = jnp.concatenate([pred_fn(ins) for ins in x_batches], axis=0)
       return y_pred
    
    def get_rmse(self, batch, batch_size=2_048):
       # Create predictions
        u, s_true = batch
        if batch_size is None: # single forward pass
          s_pred = self.apply(self.params, u)
        else: # breaks prediction into smaller forward passes
          s_pred = self.batched_apply(u, batch_size=batch_size)
        error = s_pred - s_true
        rmse = jnp.sqrt(jnp.mean(error**2))
        return rmse

    def plot_predictions(self, batch, return_pred=False, batch_size=2_048,
                         num_levels = 500):
        """Computes and plots model predictions for a given batch of data.

        Args:
            batch: data for creating/plotting results.
            return_pred: whether to return predictions after plotting 
                (default: False).
            batch_size: batch size for computations (to avoid OoM issues in the
                case of large datasets). (default: 2048)
            num_levels: number of levels for contour plot (default: 500).
        """
        # Create predictions
        u, s_true = batch
        if batch_size is None: # single forward pass
          s_pred = self.apply(self.params, u)
        else: # breaks prediction into smaller forward passes
          s_pred = self.batched_apply(u, batch_size=batch_size)

        error = s_pred - s_true
        rel_l2_error = jnp.sqrt(jnp.sum(error**2)/jnp.sum(s_true**2))
        print('Relative L2 error: {:.2e}'.format(rel_l2_error))
        print('RMSE: {:.2e}'.format(jnp.sqrt(jnp.mean(error**2))))

        plt.figure(figsize=(16, 4))

        # Ploting examples of reconstructions
        plt.subplot(131)
        plt.tricontourf(u[:,1], u[:,0],
                        s_pred.T.squeeze(), levels=num_levels, cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Prediction')

        # Ploting true solution
        plt.subplot(132)
        plt.tricontourf(u[:,1], u[:,0],
                        s_true.T.squeeze(), levels=num_levels, cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('True')

        # Ploting absolute
        plt.subplot(133)
        plt.tricontourf(u[:,1], u[:,0],
                        abs(s_pred-s_true).T.squeeze(),
                        levels=num_levels, cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Absolute Error')

        plt.show()

        plt.show()

        if return_pred:
           return s_pred