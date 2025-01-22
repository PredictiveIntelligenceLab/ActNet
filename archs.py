# Basic Library Imports
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit

from flax import linen as nn

from typing import Any, Callable, Sequence, Tuple, Union

# acceptable types for matmul precision in JAX
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]
# acceptable type for vector shapes
Shape = Sequence[int]

# identity function
identity = lambda x : x


######################################################
#################### Initializers ####################
######################################################

# Siren Initialization
def siren_initializer(key, shape, dtype=jnp.float32):
  """
  Returns a random vector of desired shape using Siren's initialization.

  Args:
    key: a PRNG key used as the random key.
    shape: shape of weights. 
    dtype: the dtype of the weights.

  Returns:
    A random Siren weight array with the specified shape and dtype.
  """
  aux = jnp.sqrt(6. / shape[0])
  return random.uniform(key, shape=shape, minval=-aux, maxval=aux, dtype=dtype)

def siren_first_layer_initializer(key, shape, dtype):
  """
  Returns a random vector of desired shape using Siren's initialization for the
  first layer.

  Args:
    key: a PRNG key used as the random key.
    shape: shape of weights. 
    dtype: the dtype of the weights.

  Returns:
    A random Siren weight array (first layer) with the specified shape & dtype.
  """
  aux = 1/shape[0]
  return random.uniform(key, shape, minval=-aux, maxval=aux, dtype=dtype)

# Custom Initialization
def kan_initializer(key, shape, dtype=jnp.float32, sigma_0=0.1):
  """
  Returns a random vector of desired shape using KAN's initialization.

  Args:
    key: a PRNG key used as the random key.
    shape: shape of weights. 
    dtype: the dtype of the weights.
    sigma (float): sigma parameter for initialization as specified in KAN paper.

  Returns:
    A random KAN weight array with the specified shape and dtype.
  """
  aux = sigma_0/jnp.sqrt(shape[0])
  return aux*random.normal(key, shape=shape, dtype=dtype)

def get_kan_initializer(sigma=0.1):
  """
  Returns a KAN initializer with desired choice of sigma.

  Args:
    sigma (float): sigma parameter for initialization as specified in KAN paper.

  Returns:
    A KAN initializer function.
  """
  return lambda key, shape, dtype=jnp.float32 : \
      kan_initializer(key, shape, dtype=dtype, sigma_0=sigma)


###############################################################
######################## Architectures ########################
###############################################################

#############
#### MLP ####
#############

class MLP(nn.Module):
  """A Multi-Layer Prerception network.

  Attributes:
    features: sequence of int detailing width of each layer.
    activation: activation function to be used in between layers (default:
      nn.gelu).
    output_activation: activation for last layer of network (default: identity).
    precision: numerical precision of the computation. See ``jax.lax.Precision``
      for details. (default: None)
  """
  features: Sequence[int]
  activation : Callable=nn.gelu
  output_activation : Callable=identity
  precision: PrecisionLike = None

  @nn.compact
  def __call__(self, x):
    """Forward pass of a MLP network.

    Args:
      x: The nd-array to be transformed.

    Returns:
      The transformed input x.
    """
    for feat in self.features[:-1]:
      x = self.activation(nn.Dense(feat, precision=self.precision)(x))
    x = nn.Dense(self.features[-1], precision=self.precision)(x) 
    return self.output_activation(x) # different activation on output layer
  

###############
#### Siren ####
###############

# see https://arxiv.org/abs/2006.09661 for details about Siren, which is an MLP
# with sine activation and a specific initialization pattern. See below for an
# iteractive colab notebook provided by the authors:
# https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb

class Siren(nn.Module):
  """A Siren network.

  Attributes:
    features: sequence of int detailing width of each layer.
    w0: frequency content parameter for mutiplying initial inputs. See Siren
      paper for more details.
    output_activation: activation for last layer of network (default: identity).
    precision: numerical precision of the computation. See ``jax.lax.Precision``
      for details. (default: None)
  """
  features: Sequence[int]
  w0 : float
  output_activation : Callable=identity
  precision: PrecisionLike = None

  @nn.compact
  def __call__(self, x):
    """Forward pass of a Siren network.

    Args:
      x: The nd-array to be transformed.

    Returns:
      The transformed input x.
    """
    x = x*self.w0
    x = jnp.sin(nn.Dense(self.features[0],
                         kernel_init=siren_first_layer_initializer,
                         precision=self.precision)(x))
    for feat in self.features[1:-1]:
      x = jnp.sin(nn.Dense(feat,
                           kernel_init=siren_initializer,
                           precision=self.precision)(x))
    x = nn.Dense(self.features[-1])(x)
    return self.output_activation(x)


################
#### ActNet ####
################

# from https://www.wolframalpha.com/input?i=E%5B%28sin%28wx%2Bp%29%29%5D+where+x+is+normally+distributed
def _mean_transf(mu, sigma, w, p):
    """ Mean of the R.V. Y=sin(w*X+p) when X is normally distributed with mean mu
    and standard deviation sigma.

    Args:
        mu: mean of the R.V. X.
        sigma: standard deviation of the R.V. X.
        w: frequency of the sinusoidal transformation.
        p: phase of the sinusoidal transformation.

    Returns:
        The mean of the transformed R.V. Y.
    """
    return jnp.exp(-0.5* (sigma*w)**2) * jnp.sin(p + mu*w)

# from https://www.wolframalpha.com/input?i=E%5Bsin%28wx%2Bp%29%5E2%5D+where+x+is+normally+distributed
def _var_transf(mu, sigma, w, p):
    """ Variance of the R.V. Y=sin(w*X+p) when X is normally distributed with
    mean mu and standard deviation sigma.

    Args:
        mu: mean of the R.V. X.
        sigma: standard deviation of the R.V. X.
        w: frequency of the sinusoidal transformation.
        p: phase of the sinusoidal transformation.

    Returns:
        The variance of the transformed R.V. Y.
    """
    return 0.5 - 0.5*jnp.exp(-2 * ((sigma*w)**2))*jnp.cos(2*(p+mu*w)) \
        -_mean_transf(mu, sigma, w, p)**2

class ActLayer(nn.Module):
    """A ActLayer module. 
    
    For further details on standard choices of initializers, please refer to
    Appendix D of the paper: https://arxiv.org/pdf/2410.01990

    Attributes:
        out_dim: output dimension of ActLayer.
        num_freqs: number of frequencies/basis functions of the ActLayer.
        use_bias: whether to add bias the the output (default: True).
        freqs_init: initializer for basis function frequencies.
        phases_init: initializer for basis function phases.
        beta_init: initializer for beta parameter.
        lamb_init: initializer for lambda parameter.
        bias_init: initializer for bias parameter.
        freze_basis: whether to freeze gradients passing thorough basis
            functions (default: False).
        freq_scaling: whether to scale basis functions to ensure mean 0 and
            standard deviation 1 (default: True).
        freq_scaling_eps: small epsilon added to the denominator of frequency
            scaling for numerical stability (default: 1e-3).
        precision: numerical precision of the computation. See
        ``jax.lax.Precision`` for details. (default: None)
    """
    out_dim : int
    num_freqs : int
    use_bias : bool=True
    # parameter initializers
    freqs_init : Callable=nn.initializers.normal(stddev=1.)  # normal w/ mean 0
    phases_init : Callable=nn.initializers.zeros
    beta_init : Callable=nn.initializers.variance_scaling(1.,
                                                          'fan_in',
                                                          distribution='uniform')
    lamb_init : Callable=nn.initializers.variance_scaling(1.,
                                                          'fan_in',
                                                          distribution='uniform')
    bias_init : Callable=nn.initializers.zeros
    # other configurations
    freeze_basis : bool=False
    freq_scaling : bool=True
    freq_scaling_eps : float=1e-3
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        """Forward pass of an ActLayer.

        Args:
            x: The nd-array to be transformed.

        Returns:
            The transformed input x.
        """
        # x should initially be shape (batch, d)

        # initialize trainable parameters
        freqs = self.param('freqs',
                           self.freqs_init,
                           (1,1,self.num_freqs)) # shape (1, 1, num_freqs)
        phases = self.param('phases',
                            self.phases_init,
                            (1,1,self.num_freqs)) # shape (1, 1, num_freqs)
        beta = self.param('beta',
                          self.beta_init,
                          (self.num_freqs, self.out_dim)) # shape (num_freqs, out_dim)
        lamb = self.param('lamb',
                          self.lamb_init,
                          (x.shape[-1], self.out_dim)) # shape (d, out_dim)

        if self.freeze_basis:
            freqs = jax.lax.stop_gradient(freqs)
            phases = jax.lax.stop_gradient(phases)
        
        # perform basis expansion
        x = jnp.expand_dims(x, 2) # shape (batch, d, 1)
        x = jnp.sin(freqs*x + phases) # shape (batch_dim, d, num_freqs)
        if self.freq_scaling:
            x = (x - _mean_transf(0., 1., freqs, phases)) \
                / (jnp.sqrt(self.freq_scaling_eps + _var_transf(0., 1.,
                                                                freqs, phases)))

        
        # efficiently computes eq 6 from https://arxiv.org/pdf/2410.01990 using
        # einsum. Depending on hardware and JAX/CUDA version, there may be
        # slightly faster alternatives, but we chose this one for the sake of
        # simplicity/clarity.
        x = jnp.einsum('...ij, jk, ik-> ...k', x, beta, lamb,
                       precision=self.precision)

        # optionally add bias
        if self.use_bias:
           bias = self.param('bias',
                             self.bias_init,
                             (self.out_dim,))
           x = x + bias # Shape (batch_size, out_dim)

        return x # Shape (batch_size, out_dim)
    

class ActNet(nn.Module):
    """A ActNet module.

    Attributes:
        embed_dim: embedding dimension for ActLayers.
        num_layers: how many intermediate blocks are used.
        out_dim: output dimension of ActNet.
        num_freqs: number of frequencies/basis functions of the ActLayers.
        output_activation: output_activation: activation for last layer of
            network (default: identity).
        op_order: order of operations contained in each intermediate block. This
            should be a string containing only 'A' (ActLayer), 'S' (Skip
            connection) or 'L' (LayerNorm) characters. (default: 'A')
        use_act_bias: whether to add bias the the output of ActLayers
            (default: True).
        freqs_init: initializer for basis function frequencies of ActLayers.
        phases_init: initializer for basis function phases of ActLayers.
        beta_init: initializer for beta parameter of ActLayers.
        lamb_init: initializer for lambda parameter of ActLayers.
        act_bias_init: initializer for bias parameter of ActLayers.
        proj_bias_init: initializer for bias parameter of initial projection
            Layer.
        w0_init: initializer for w0 scale parameter.
        w0_fixed: if False, initializes w0 using w0_init. Otherwise uses given
            fixed w0 (default: False).
        freze_basis: whether to freeze gradients passing thorough basis
            functions (default: False).
        freq_scaling: whether to scale basis functions to ensure mean 0 and
            standard deviation 1 (default: True).
        freq_scaling_eps: small epsilon added to the denominator of frequency
            scaling for numerical stability (default: 1e-3).
        precision: numerical precision of the computation. See
        ``jax.lax.Precision`` for details. (default: None)
    """
    embed_dim : int
    num_layers : int # number of layers in the network
    out_dim : int # dimension of output vector
    num_freqs : int # how many frequencies/basis functions to use in ActLayers
    output_activation : Callable = identity
    op_order : str='A' 
    # op_order should be a string containing only 'A' (ActLayer), 'S' (Skip
    # connection) or 'L' (LayerNorm) characters. This feature was used for
    # development/debugging, but is not used in any experiment of the paper.

    # parameter initializers
    freqs_init : Callable=nn.initializers.normal(stddev=1.)  # normal w/ mean 0
    phases_init : Callable=nn.initializers.zeros
    beta_init : Callable=nn.initializers.variance_scaling(1., 'fan_in',
                                                          distribution='uniform')
    lamb_init : Callable=nn.initializers.variance_scaling(1., 'fan_in',
                                                          distribution='uniform')
    act_bias_init : Callable=nn.initializers.zeros
    proj_bias_init : Callable=lambda key, shape, dtype :\
        random.uniform(key, shape, dtype,
                       minval=-jnp.sqrt(3), maxval=jnp.sqrt(3))
    
    w0_init : Callable=nn.initializers.constant(30.) # following SIREN strategy
    w0_fixed : Union[False, float]=False # if False, initializes w0 as above. Otherwise uses given fixed w0

    # other ActLayer configurations
    use_act_bias : bool=True
    freeze_basis : bool=False
    freq_scaling : bool=True
    freq_scaling_eps : float=1e-3
    precision: PrecisionLike = None # numerical precision for matrix operations

    @nn.compact
    def __call__(self, x):
        """Forward pass of an ActNet.

        Args:
            x: The nd-array to be transformed.

        Returns:
            The transformed input x.
        """
        # initialize w0 parameter
        if self.w0_fixed is False:
            # trainable scalar parameter
            w0 = self.param('w0',
                            self.w0_init,
                            ())
            # use softplus to ensure w0 is positive and does not decay to zero
            # too fast (used only while debugging).
            w0 = nn.softplus(w0)
        else: # use user-specified value for w0
            w0 = self.w0_fixed
        # scale by w0 factor, then project to embeded dimension
        x = x*w0
        x = nn.Dense(self.embed_dim, bias_init=self.proj_bias_init,
                     precision=self.precision)(x)
        
        for _ in range(self.num_layers):
            y = x # store initial value as x, do operations on y
            for char in self.op_order:
                if char == 'A': # ActLayer
                    y  = ActLayer(
                            out_dim = self.embed_dim,
                            num_freqs = self.num_freqs,
                            use_bias = self.use_act_bias,
                            freqs_init = self.freqs_init,
                            phases_init = self.phases_init,
                            beta_init = self.beta_init,
                            lamb_init = self.lamb_init,
                            bias_init = self.act_bias_init,
                            freeze_basis = self.freeze_basis,
                            freq_scaling = self.freq_scaling,
                            freq_scaling_eps = self.freq_scaling_eps,
                            precision=self.precision,
                            )(y)
                elif char == 'S': # Skip connection
                    y = y + x
                elif char == 'L': # LayerNorm
                    y = nn.LayerNorm()(y)
                else:
                    raise NotImplementedError(f"Could not recognize option '{char}'. Options for op_order should be 'A' (ActLayer), 'S' (Skip connection) or 'L' (LayerNorm).")
            x = y # update value of x after all operations are done

        # project to output dimension and potentially use output activation
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.he_uniform(),
                     precision=self.precision)(x)
        x = self.output_activation(x)

        return x


##############
#### KAN #####
##############

# Adapted to JAX from the "EfficientKAN" GitHub repository (PyTorch). Code was
# altered as little as possible, for the sake of consistency/fairness.
# https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py

class KANLinear(nn.Module):
    in_features : int
    out_features : int
    grid_size : int=5
    spline_order: int=3
    scale_noise : float=0.1
    scale_base : float=1.0
    scale_spline : float=1.0
    enable_standalone_scale_spline : bool=True
    base_activation : Callable=nn.silu
    grid_eps : float=0.02
    grid_range : Sequence[Union[float, int]]=(-1,1)
    precision: PrecisionLike = None

    def setup(self):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        self.h = h
        grid = (
            (
                jnp.arange(start=-self.spline_order, stop=self.grid_size + self.spline_order + 1) * h
                + self.grid_range[0]
            )
        )
        self.grid = grid * jnp.ones((self.in_features, 1))

        self.base_weight = self.param('base_weight', # parameter name
                                      nn.initializers.he_uniform(), # initialization funciton
                                      (self.out_features, self.in_features)) # shape info
        self.spline_weight = self.param('spline_weight', # parameter name
                                        nn.initializers.he_uniform(), # initialization funciton
                                        (self.out_features, self.in_features, self.grid_size+self.spline_order)) # shape info

        if self.enable_standalone_scale_spline:
            self.spline_scaler = self.param('spline_scaler', # parameter name
                                            nn.initializers.he_uniform(), # initialization funciton
                                            (self.out_features, self.in_features)) # shape info
            

    def b_splines(self, x: jax.Array):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert len(x.shape) == 2 and x.shape[1] == self.in_features

        # grid is shape (in_features, grid_size + 2 * spline_order + 1)
        grid = self.grid
        x = jnp.expand_dims(x, -1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:]))
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.shape == (
            x.shape[0],
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            jnp.expand_dims(self.spline_scaler, -1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def __call__(self, x: jax.Array):
        assert x.shape[-1] == self.in_features, f"x.shape[-1]={x.shape[-1]} should be equal to {self.in_features}"
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = jnp.matmul(self.base_activation(x), self.base_weight.T, precision=self.precision)
        spline_output = jnp.matmul(
            self.b_splines(x).reshape(x.shape[0], -1),
            self.scaled_spline_weight.reshape(self.out_features, -1).T,
            precision=self.precision,
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output
    

class KAN(nn.Module):
    features : Sequence[int]
    output_activation : Callable=identity
    grid_size : int=5
    spline_order: int=3
    scale_noise : float=0.1
    scale_base : float=1.0
    scale_spline : float=1.0
    enable_standalone_scale_spline : bool=True
    base_activation : Callable=nn.silu
    grid_eps : float=0.02
    grid_range : Sequence[Union[float, int]]=(-1,1)
    precision: PrecisionLike = None

    def setup(self):
        self.layers = [KANLinear(
            self.features[i],
            self.features[i+1],
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            scale_noise=self.scale_noise,
            scale_base=self.scale_base,
            scale_spline=self.scale_spline,
            enable_standalone_scale_spline=self.enable_standalone_scale_spline,
            base_activation=self.base_activation,
            grid_eps=self.grid_eps,
            grid_range=self.grid_range,
            precision=self.precision,
                                 ) for i in range(len(self.features) - 1)]
        
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return self.output_activation(x)
    


############################################################
################### Architecture Builder ###################
############################################################

def arch_from_config(arch_config):
    ''' Given a config file, outputs architecture object with given
    configurations.

    Args:
        arch_config: config file specifying architecture hyperparameters.

    Returns:
        Architecture as a Flax Linen nn.Module.
    '''
    if arch_config.arch_type == 'ActNet':
        arch = ActNet(**arch_config.hyperparams)
        return arch
    elif arch_config.arch_type == 'MLP':
        arch = MLP(**arch_config.hyperparams)
        return arch
    elif arch_config.arch_type == 'Siren':
        arch = Siren(**arch_config.hyperparams)
        return arch
    elif arch_config.arch_type == 'KAN':
        arch = KAN(**arch_config.hyperparams)
        return arch
    else:
        raise NotImplementedError(f"Cannot recognize arch_type {arch_config.arch_type}. So far, only 'ActNet', 'MLP', 'Siren' and 'KAN' are implemented")