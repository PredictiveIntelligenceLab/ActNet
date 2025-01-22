# Basic Library Importsk
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, value_and_grad

import optax
from optax._src import linear_algebra
import jaxopt

import matplotlib.pyplot as plt

from functools import partial
import itertools
from tqdm import trange

import sys
sys.path.append('..') # makes modules in parent repository available to import
from models import plot_logs



def get_laplacian(f):
    '''Computes the 2D laplacian of a given funciton in a computationally
    efficient way by avoiding unecessary repetition of the computational graph.

    Args:
        f (Callable): a function with signature (params, x, y)->scalar
    Returns:
        (Callable): a function with signature (params, x, y)->scalar that is the
                    laplacian of input function
    '''
    _single_f = lambda params, x, y : f(params, x[None,:], y[None,:]).squeeze()
    def _scalar_grads(params, x, y):
        ux, uy = grad(_single_f, argnums=(1,2))(params, x, y)
        return ux.squeeze(), uy.squeeze()
    def _lapl_aux(params, x, y):
        (u_x, u_y), u_xx =  value_and_grad(_scalar_grads,
                                           argnums=1,
                                           has_aux=True)(params, x, y)
        return u_y, u_xx.squeeze()
    def _lapl_aux_2(params, x, y):
        (u_y, u_xx), u_yy =  value_and_grad(_lapl_aux,
                                            argnums=2,
                                            has_aux=True)(params, x, y)
        return u_xx, u_yy.squeeze()
    def _lapl(params, xs, ys):
        uxx, uyy = vmap(_lapl_aux_2, in_axes=(None, 0, 0))(params, xs, ys)
        return uxx + uyy
    return _lapl

class PoissonModel:
    """ A model for training/evaluating a neural network using physics-informed
    training on the 2D Poisson problem.

    Attributes:
        arch (nn.Module): a Flax module of the desired architecute.
        batch: an initial batch used for initializing parameters and computing
            normalization factors.
        true_fun: the true ground_truth function, if known (default: None).
        pde_res_fn: the target function for the PDE differential operator.
        optimizer: the optimizer to be used when running gradient descent.
        normalize_inputs: whether to normalize inputs before passing them to the
            architecture (default: True).
        key: a PRNG key used as the random key for initialization.
        pinn_weights: pair of weights for balacing residual/border losses.
        exact_bd_condition: whether to exactly enforce boundary condition on 
            [-1,1]^2 (see  https://arxiv.org/abs/2104.08426). This implicitly
            makes it so that border losses are not computed (defaut: True).
        bdr_enforcer_order: order of the polynomial used for enforcing border 
            exactly. Should be an even integer (default: 2).
        steps_per_check (int): how many training steps to use between logging
            and displaying losses (default: 50).
    """
    def __init__(self, arch, batch, true_fun=None,
                 pde_res_fn=lambda x, y : \
                    -(jnp.pi**2)*(1+4*(y**2))*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*(y**2)) \
                        + 2*jnp.pi*jnp.sin(jnp.pi*x)*jnp.cos(jnp.pi*(y**2)),
                 optimizer=None, normalize_inputs=True, key=random.PRNGKey(43),
                 pinn_weights=(0.001, 1.), exact_bd_condition=True,
                 bdr_enforcer_order=2, steps_per_check=50) -> None:
        # Define model
        self.arch = arch
        self.key = key
        self.steps_per_check = steps_per_check
        self.pde_res_fn = pde_res_fn
        self.true_fun = true_fun
        self.pinn_weights = pinn_weights # not really used in the experiments (boundary is enforced exactly)

        # Initialize parameters
        interior_batch, border_batch = batch
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
               _apply = lambda params, x, y : \
                (1-x**self.bdr_enforcer_order)\
                    *(1-y**self.bdr_enforcer_order)\
                    *self.arch.apply(params, (jnp.hstack([x, y])-mu_x)/sig_x)
            else:
               _apply = lambda params, x, y : \
                self.arch.apply(params, (jnp.hstack([x, y])-mu_x)/sig_x)
        else:
            self.norm_stats = None
            if self.exact_bd_condition:
               _apply = lambda params, x, y : \
                (1-x**self.bdr_enforcer_order)\
                    *(1-y**self.bdr_enforcer_order)\
                    *self.arch.apply(params, jnp.hstack([x, y]))
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
        self.laplacian = get_laplacian(self.apply)

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

        # Optimizer LBFGS
        self.optimizer_lbfgs = jaxopt.LBFGS(self.loss)
        self.opt_state_lbfgs = self.optimizer_lbfgs.init_state(self.params,
                                                               batch)
        self.optimizer_update_lbfgs = jit(self.optimizer_lbfgs.update)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.grad_norm_log = []
        self.rel_l2_log = []
    
    def residual_loss(self, params, x, y):
        res = self.laplacian(params, x, y)[:,None] # shape (batch_dim,1)
        goal = self.pde_res_fn(x, y) # shape (batch_dim,1)
        return jnp.mean((res-goal)**2, axis=-1) # shape (batch_dim,)
    
    def border_loss(self, params, x, y):
        # function should be zero at the boundary
        outputs = self.apply(params, x, y) # shape (batch_dim, out_dim)
        return jnp.mean(outputs**2, axis=-1) # shape (batch_dim,)

    def pinn_loss(self, params, interior_batch, border_batch):
        r_loss = self.residual_loss(params,
                                    interior_batch[:,0][:,None],
                                    interior_batch[:,1][:,None])
        if self.exact_bd_condition:
            # no need to consider border loss, since it will be 0 when bdry
            # condition is exactly enforced
            return self.pinn_weights[0]*r_loss.mean()
        else:
            # consider both residual loss and boundary condition loss
            b_loss = self.border_loss(params,
                                      border_batch[:,0][:,None],
                                      border_batch[:,1][:,None])
            return self.pinn_weights[0]*r_loss.mean() \
                + self.pinn_weights[1]*b_loss.mean()    

    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        interior_batch, border_batch = batch
        return self.pinn_loss(params, interior_batch, border_batch).mean() # scalar
    

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, batch):
        grads = grad(self.loss)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grads

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 10_000):
        """ Trains the neural network for nIter steps using data loader.

        Args:
            dataset (SquareDataset): data loader for training.
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
                if self.true_fun is not None: # true function is known
                    interior_batch, border_batch = batch
                    pred = self.apply(self.params,
                                      interior_batch[:,0][:,None],
                                      interior_batch[:,1][:,None])
                    true = self.true_fun(interior_batch[:,0][:,None],
                                         interior_batch[:,1][:,None])
                    rel_l2_error = jnp.sqrt(((pred-true)**2).mean() \
                                            / ((true)**2).mean())
                    self.rel_l2_log.append(rel_l2_error)
                    pbar.set_postfix_str(
                        f"loss:{l : .3e}, rel_l2:{rel_l2_error : .2e}, 'grad_norm':{jnp.mean(jnp.array(g_norm)) : .2e}")
                else: # true function is unknown
                    pbar.set_postfix({
                        'loss': l,
                        'grad_norm': jnp.mean(jnp.array(g_norm))})

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
            dataset (SquareDataset): data loader for training.
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
                if self.true_fun is not None:
                    interior_batch, border_batch = batch
                    pred = self.apply(self.params, interior_batch[:,0][:,None],
                                      interior_batch[:,1][:,None])
                    true = self.true_fun(interior_batch[:,0][:,None],
                                         interior_batch[:,1][:,None])
                    rel_l2_error = jnp.sqrt(((pred-true)**2).mean() \
                                            / ((true)**2).mean())
                    self.rel_l2_log.append(rel_l2_error)
                    pbar.set_postfix_str(
                        f"loss:{l : .3e}, rel_l2:{rel_l2_error : .2e}, 'grad_norm':{jnp.mean(jnp.array(g_norm)) : .2e}")
                else:
                    pbar.set_postfix({
                        'loss': l,
                        'grad_norm': jnp.mean(jnp.array(g_norm))})
            # take step
            self.params, self.opt_state_lbfgs = self.step_lbfgs(self.params,
                                                                self.opt_state_lbfgs,
                                                                batch)
            
    
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
       pred_fn = jit(lambda ins : self.apply(self.params,
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
                         num_levels = 100):
        """Computes and plots model predictions for a given batch of data.

        Args:
            batch: data for creating/plotting results.
            return_pred: whether to return predictions after plotting 
                (default: False).
            batch_size: batch size for computations (to avoid OoM issues in the
                case of large datasets). (default: 2048)
            num_levels: number of levels for contour plot (default: 100).
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
        plt.tricontourf(u[:,0], u[:,1],
                        s_pred.squeeze(), levels=num_levels)
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Prediction')

        # Ploting true solution
        plt.subplot(132)
        plt.tricontourf(u[:,0], u[:,1],
                        s_true.squeeze(), levels=num_levels)
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('True')

        # Ploting absolute
        plt.subplot(133)
        plt.tricontourf(u[:,0], u[:,1],
                        abs(s_pred-s_true).squeeze(), levels=num_levels)
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Absolute Error')

        plt.show()

        plt.show()

        if return_pred:
           return s_pred

