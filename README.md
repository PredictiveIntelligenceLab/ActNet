# ActNet

Repository for some of the experiments presented in the paper "Deep Learning Alternatives of the Kolmogorov Superposition Theorem", acepted as a Spotlight Paper in ICLR 2025. (arXiv: https://arxiv.org/abs/2410.01990)

This code requires common libraries of the JAX environment, such as Flax (for neural network design) and Optax/JaxOpt (for training and optimization). Plotting is done using Matplotlb.

Experiments comparing against the state-of-the-art require integration with JaxPi, which is an open-source library. The code for those experiments can now be found on the ActNet branch of JaxPi: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/ActNet

FILES:
* archs.py : includes the architectures used in the paper, including JAX implementations of ActNet, KAN and Siren.
* models.py : includes a training model for regression that can be used with any of the architectures.
* utils.py : includes useful code for sampling batches.
* poisson_2d/ : directory containing minimal code to run the Poisson 2D problem.
    * PoissonModel.py : flexible training model for the 2D Poisson problem that can be used with any desired architecture.
    * prediction_plots.ipynb : Jupyter notebook tutorial showing how to run the Poisson problem and produce plots.
* helmholtz_2d/ : directory containing minimal code to run the Helmholtz 2D problem.
    * HelmholtzModel.py : flexible training model for the 2D Helmholtz problem that can be used with any desired architecture.
    * prediction_plots.ipynb : Jupyter notebook tutorial showing how to run the Helmholtz problem and produce plots.
* allen_cahn/ : directory containing minimal code to run the Allen-Cahn problem.
    * ac_solution_dirichlet.pkl : pickle file containing reference solution for the Allen-Cahn obtained using a classical solver.
    * CausalAllenCahnModel.py : flexible training model for the Allen-Cahn problem that can be used with any desired architecture.
    * prediction_plots.ipynb : Jupyter notebook tutorial showing how to run the Allen-Cahn problem and produce plots.
