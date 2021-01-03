# Batch Test for COVID-19

This repository explores the techniques developed in:
>[Modeling and Computation of High Efficiency and Efficacy Multi-Step Batch Testing for Infectious Diseases] (https://arxiv.org/abs/2006.16079)
Hongshik Ahn, Haoran Jiang, Xiaolin Li

## Table of Contents
1. Installation
2. Project Motivation
3. Code and Notebook

## Installation

This project requires **Python 3.x** and the following Python libraries installed:

* Scipy
* Numpy
* Pandas
* matplotlib
* scikit-learn
* Numba

Installation is most easily done by using pip.
1. Create or activate a virtual environment (e.g. using `virtualenv` or `conda`)
2. Install required packages
```bash
cd <your directory>
git clone https://github.com/Haoran-Jiang/batchtest_covid19
cd batchtest_covid19
pip install --ignore-installed -r requirements.txt
```

You will also need to have software installed to run and execute an iPython Notebook

We recommend you install Anaconda, a pre-packaged Python distribution that contains most of the necessary libraies and software for this project.

## Project Motivation

We propose a mathematical model based on probability theory to optimize COVID-19 testing
by a multi-step batch testing approach with variable batch sizes. This model and simulation tool
dramatically increase the efficiency and efficacy of the tests in a large population at a low cost,
particularly when the infection rate is low. The proposed method combines statistical modeling
with numerical methods to solve nonlinear equations and obtain optimal batch sizes at each step
of tests, with the flexibility to incorporate geographic and demographic information. In theory,
this method substantially improves the false positive rate and positive predictive value as well.
We also conducted a Monte Carlo simulation to verify this theory. Our simulation results show
that our method significantly reduces the false negative rate. More accurate assessment can be
made if the dilution effect or other practical factors are taken into consideration. The proposed
method will be particularly useful for the early detection of infectious diseases and prevention of
future pandemics. The proposed work will have broader impacts on medical testing for contagious
diseases in general.

## Code and Notebook

All necessary code is contained in `fast_btk.py`. We have one notebook showing how to run our code and reproduce results.
## Citation

```
@misc{ahn2020modeling,
      title={Modeling and Computation of High Efficiency and Efficacy Multi-Step Batch Testing for Infectious Diseases}, 
      author={Hongshik Ahn and Haoran Jiang and Xiaolin Li},
      year={2020},
      eprint={2006.16079},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```
