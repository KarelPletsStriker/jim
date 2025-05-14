# Jim <img src="https://user-images.githubusercontent.com/4642979/218163532-1c8a58e5-6f36-42de-96d3-f245eee93cf8.png" alt="jim" width="35"/> - A JAX-based gravitational-wave inference toolkit

<a href="https://jim.readthedocs.io/en/main/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>

Jim comprises a set of tools for estimating parameters of gravitational-wave sources thorugh Bayesian inference.
At its core, Jim relies on the JAX-based sampler [flowMC](https://github.com/kazewong/flowMC),
which leverages normalizing flows to enhance the convergence of a gradient-based MCMC sampler.

Since its based on JAX, Jim can also leverage hardware acceleration to achieve significant speedups on GPUs. Jim also takes advantage of likelihood-heterodyining, ([Cornish 2010](https://arxiv.org/abs/1007.4820), [Cornish 2021](https://arxiv.org/abs/2109.02728)) to compute the gravitational-wave likelihood more efficiently.

See the accompanying paper, [Wong, Isi, Edwards (2023)](https://github.com/kazewong/TurboPE/) for details.


> [!WARNING]  
> Jim is under heavy development, so API is constantly changing. Use at your own risk!
> One way to mitigate this inconvience is to make your own fork over a version for now.
> We expect to hit a stable version this year. Stay tuned.

_[Documentatation and examples are a work in progress]_

## Installation

In order to install this forked version of JIM with the `TriangularGroundBased3G` and `SpaceBased` Detector classes.

```
pip install git+[my own link will have to go here]
pip install flowMC==0.3.4 evosax==0.1.5
```

If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

_NOTE:_ Jim is only currently compatible with Python 3.10.

## Performance

The performance of Jim will vary depending on the hardware available. Under optimal conditions, the CUDA installation can achieve parameter estimation in ~1 min on an Nvidia A100 GPU for a binary neutron star (see [paper](https://github.com/kazewong/TurboPE/) for details). If a GPU is not available, JAX will fall back on CPUs, and you will see a message like this on execution:

```
No GPU/TPU found, falling back to CPU.
```

## Directory

Parameter estimation examples are in `example/ParameterEstimation`.

## Attribution

Please cite the accompanying paper, [Wong, Isi, Edwards (2023)](https://github.com/kazewong/TurboPE/).

# Changes made for the Space Based Detector Class

In preparation of the Space detector class, two new python packages need to be installed additionally:  [`fastlisaresponse`](https://github.com/mikekatz04/lisa-on-gpu/tree/master) and [`lisatools`](https://github.com/mikekatz04/LISAanalysistools), managed by Michael Katz.

Their installation is as follows (done after the `JIM` installation)

## Installing the LISA packages
First load the right modules (only relevant for users of VSC-HPC@KULeuven)
```
module purge
module load CUDA/12 GCC/11.3.0 GSL FFTW/3.3.10-GCC-11.3.0
```
Then install the following packages via pip:

```
pip install multispline pygments matplotlib Cython numpy scipy h5py requests ipython astropy
pip install cupy-cuda12x
pip install lisaanalysistools
```

Finally, you should install the newest version of `fastlisaresponse` by copying it from git:
```
git clone https://github.com/mikekatz04/lisa-on-gpu.git
cd lisa-on-gpu
python scripts/prebuild.py
pip install .
```
