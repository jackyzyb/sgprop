# Synthetic-Gradient Expert Simulation

This directory implements the finite-partition expert-network experiment, as described in Section 6 and Section H.1 from the paper. The experiment in paper uses default parameters:

| Parameter | Value |
| --- | --- |
| Dimension $d$ | $4,5,\ldots,12$ |
| Projection size $k$ | $2,3,4$ |
| Experts $m$ | $5$ |
| Observations per state $n$ | $1,2,3,4$ |
| Monte Carlo trials | $20$ |
| $\sigma, p, \tau$ | $0.5, 0.5, 0.5$ |
| Bootstrap resamples for ratio intervals | $500$ |
---


The experiment requires Python with `numpy` and `matplotlib` installed. 

To run the experiment and get the results:

```bash
python experiment.py
```
