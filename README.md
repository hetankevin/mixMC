
# Learning Mixtures of Continuous-Time Markov Chains

This repository contains code to learn mixtures of continuous-time (and
discrete-time) Markov chains using methods. Our method is split into three steps:

1. Discretization
2. Soft-Clustering: We use methods from learning discrete-time Markov chains
    - Discrete-Time EM
    - SVD (Gupta et al., Spaeh and Tsourakakis)
    - Spectral Clustering (Kausik et al.)
3. Recovery

Additionally, we provide code for continuous-time EM. 

Methods are split into the representation of discrete and continuous-time
mixtures ([dtmixtures.py](dtmixtures.py), [ctmixtures.py](ctmixtures.py)) and learning of discrete and
continuous-time mixtures ([dtlearn.py](dtlearn.py), [ctlearn.py](ctlearn.py)). The code for experiments is
in [experiments.py](experiments.py).


## Running the Code

Enter the Anaconda environment:

> conda activate std_env

Compile cython:

> python setup.py build_ext --inplace

We can run a small example using

> python run.py

which places a figure under `plots`. All results are cached in the `cache`
folder. Empty the folder to re-run experiments.


## Preparing the Real-World Datasets

In order to experiment with the NBA data from Second Spectrum, put `passes.csv`
and `player_ssid.csv` in the [NBA](NBA) folder and run [NBA/extract.py](NBA/extract.py).

The LastFM data is available under http://millionsongdataset.com/lastfm/#desc
Unzip `lastfm-dataset-1K.tar.gz` in the folder [LastFM](LastFM).


## Reproducing the Experiments

Code for plots produced in the paper are in the notebook [main.ipynb](main.ipynb)

