
# Learning Mixtures of Continuous-Time Markov Chains

This repository contains code to learn mixtures of continuous-time (and
discrete-time) markov chains using methods, based on
- Discrete-Time EM
- SVD (Gupta et al.)
- Spectral Clustering (Kausik et al.)
- Continuous-Time EM

Methods are split into the representation of discrete and continuous-time
mixtures (`dtmixtures.py`, `ctmixtures.py`) and learning of discrete and
continuous-time mixtures (`dtlearn.py`, `ctlearn.py`). The code for experiments is
in `experiments.py`.


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
and `player_ssid.csv` in the `NBA`-folder and run `extract.py`.

The LastFM data is available under http://millionsongdataset.com/lastfm/#desc
Unzip `lastfm-dataset-1K.tar.gz` in the folder `LastFM`.


## Reproducing the Experiments

Code for plots produced in the paper are in the notebook `main.ipynb`

