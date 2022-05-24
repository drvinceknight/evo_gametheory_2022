# Quick Tutorial

## Setting up the virtual environment

Note that currently all code is supported to run on Python 3.8, 3.9 and 3.10.

To create a virtual environment:

    $ python -m venv env

To activate the environment:

    $ source env/bin/activate

To install the specific versions of all libraries:

    $ python -m pip install -r requirements.txt
    
## Testing

All code is formatted using [`black`](https://github.com/psf/black). To check
that all code is formatted correctly:

    $ python -m black .

## Basic example

to run, execute the following in the terminal with your desired population and weight as the terminal arguments. 

e.g. `python main.py normal pareto` will run a simulation with a population where the mass is normally distribution with $\mu = 5$ and $\sigma = 12.5$ and weight follows a pareto distribution with $\alpha = 1$ and $m = 1/10$. Of course, these values can be changedin the `vars.py` file.

This code uses the [axelrod](https://axelrod.readthedocs.io/en/stable/index.html) package to simulate Moran Processes with [heterogeneous populations](https://axelrod.readthedocs.io/en/stable/tutorials/creating_heterogenous_player_moran_process/index.html).

## Heterogeneity

Hetereogeneity is introduced by manipulation the standard payoffs:

$T = 5$

$R = 3$

$P = 1$

$S = 0$

Into

$T = 5\times m_{p_2} + m_{p_1}\times W_{p_1}$

$R = 3\times  m_{p_2} + m_{p_1}\times W_{p_1}$

$P = 1\times  m_{p_2} + m_{p_1}\times W_{p_1}$

$S = 0\times  m_{p_2} + m_{p_1}\times W_{p_1}$

where

$m_{p_n}$ denotes the mass $m$ of player $p_n$ 

and

$w_{p_n}$ denotes the weight $w$ of player $p_n$

## Distributions

`vars.py` includes a [pareto distribution](https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html) and a [truncated normal distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html). Homogenous populations are generated by passing `homo` as the first terminal argument. 

e.g.

`python main.py homo 0`

note that the weight $w$ has no effect when mass $m$ is constant.

## Strategies

The following strategies are by default generated in the population as included in `axelrod.basic_strategies`:

- Alternator
- Anti Tit For Tat
- Bully
- Cooperator
- Cycler DC
- Defector
- Suspicious Tit For Tat
- Tit For Tat
- Win-Shift Lose-Stay: D
- Win-Stay Lose-Shift

## Analysis

The analysis, performed in R can be found in the `analysis` folder and previewed [here](https://htmlpreview.github.io/?https://github.com/vocelik/evo_gametheory_2022/blob/main/analysis/main.html).
