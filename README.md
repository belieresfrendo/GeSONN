# GeSONN

Welcome on the git page of GeSONN, the open source code for GEometric Shape Optimization with Neural Networks.

## Prerequisites

Install [Git](https://about.gitlab.com/free-trial/devsecops/?utm_medium=cpc&utm_source=google&utm_campaign=brand_rlsa__global_exact&utm_content=free-trial&utm_term=git%20lab&_bt=656315922370&_bk=git%20lab&_bm=e&_bn=g&_bg=148481441276&gclid=CjwKCAjw6p-oBhAYEiwAgg2PgsbJIxXSSXyydPb8B8HdSkynh4z99dIjYXLTUDxzlizGVpjN_ipAABoCvRwQAvD_BwE), [FreeFem++ 4.13](https://freefem.org/) and [Python 3.11.5](https://www.python.org/downloads/), we suggest to work with [Conda 23.11.0](https://www.anaconda.com/download/).

## Install and setup GeSONN

This is a quick guide on how to install GeSONN and the required dependencies on your machine. GeSONN is developed on macOS machines (version 12.6.7) with recent Python versions > 3.11. These instructions assume you are familiar with working in a terminal. This guide is currently described for macOS only, but should be easily adaptable to Linux or Windows.

```bash
git clone git@github.com:belieresfrendo/GeSONN.git
cd GeSONN
```

### Create a python virtual environment

```
python3 -m venv venv_gesonn
source venv_gesonn/bin/activate
```

### Install the package

```bash
pip install -e .
```

To test that the installation is correct, run
```
python
import gesonn
```
If no error comes up, you are good to go.

## Update GeSONN

To update go to your GeSONN repository [YOURDIR]/GeSONN, pull the latest changes via:
```
git pull
```

## To contact me

Feel free to contact me at *amaury.belieres@math.unistra.fr* if you'd like to collaborate on this framework, or if you have any questions!

## Acknowledgments

* [Victor Michel-Dansac](https://irma.math.unistra.fr/~micheldansac/)
* [Yannick Privat](https://yannick-privat.perso.math.cnrs.fr/)
* [Emmanuel Frank](https://irma.math.unistra.fr/~franck/)

## To quote this project

Incoming...

<!-- 

## Launch tests

```bash
pip install -e ".[test]"
pytest
```

## Generate documentation

```bash
pip install -e ".[doc]"
cd docs
env PYTORCH_JIT=0 make html
```

html docs are generated in \_build/html
 -->
