
*A Julia package for robust ensemble filtering*


This repository is a companion to the work [^1]: Le Provost, Baptista, Eldredge, and Marzouk (2023) "An adaptive ensemble filter for heavy-tailed distributions: tuning-free inflation and localization," under preparation.


Heavy tails is a common feature of filtering distributions that results from the nonlinear dynamical and observation processes as well as the uncertainty from physical sensors. In these settings, the Kalman filter and its ensemble version --- the ensemble Kalman filter (EnKF) --- that have been designed under Gaussian assumptions result in  degraded performance. Leveraging tools from measure transport (Spantini et al., SIAM Review, 2022), we present a generalization of the EnKF whose prior-to-posterior update leads to exact inference for t-distributions. We demonstrate that this filter is less sensitive to outlying synthetic observations generated by the observation model for small $\nu$. Moreover, it recovers the Kalman filter for $\nu = \infty$. For nonlinear state-space models with heavy-tailed noise, we propose an algorithm to estimate the prior-to-posterior update from samples of joint forecast distribution of the states and observations. We rely on a regularized expectation-maximization (EM) algorithm to estimate the mean, scale matrix, and degree of freedom of heavy-tailed t-distributions from limited samples (Finegold and Drton, arXiv preprint, 2014). By sequentially estimating the degree of freedom at each analysis step, our filter has the appealing feature of  adapting the prior-to-posterior update to the tail-heaviness of the data. This new filter intrinsically embeds an adaptive and data-dependent multiplicative inflation mechanism complemented with an adaptive localization through the $l1$-penalization of the estimated scale matrix. We demonstrate the benefits of this new ensemble filter on challenging filtering problems with heavy-tailed noise.


This repository contains the source code and Jupyter notebooks to reproduce the numerical experiments and Figures in Le Provost et al. [^1]


## Installation

This package works on Julia `1.6` and above. To install from the REPL, type
e.g.,
```julia
] add https://github.com/mleprovost/Paper-Ensemble-Robust-Filter.jl.git
```

Then, in any version, type
```julia
julia> using RobustFilter
```

## Correspondence email
[mleprovo@mit.edu](mailto:mleprovo@mit.edu)

## References

[^1]: Le Provost, Baptista, Eldredge, and Marzouk (2023) "An adaptive ensemble filter for heavy-tailed distributions: tuning-free inflation and localization," under preparation.

## Licence

See [LICENSE.md](https://github.com/mleprovost/Paper-Ensemble-Robust-Filter.jl/raw/main/LICENSE.md)

