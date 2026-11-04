# LP experiment notes

See the [repository README](../../README.md#lp-results) for the reported results and basic run commands. The [timing protocol](../../results/paper_lp/README.md) describes Tables 1 and 2.

## Source files

| Setting | Source |
|---|---|
| Synthetic and Netlib case configurations | [final_projection_figure_suite.py](../core/final_projection_figure_suite.py) |
| Cost sampling | [compare_fixedX_family_suite.py](../core/compare_fixedX_family_suite.py) |
| Estimated-prior experiments | [run_subgaussian_beyond_prior_suite.py](run_subgaussian_beyond_prior_suite.py) and [run_beyond_prior_sample_sweeps_paper8.py](run_beyond_prior_sample_sweeps_paper8.py) |
| Neural projection training | [fcnn_training_protocols.py](fcnn_training_protocols.py) |

## SCAGR25 and STAIR sampling

The current sampler adds Gaussian perturbations to the objective coefficients and limits their norm to the selected sampling radius. The case settings are defined in `netlib_cases()` in [final_projection_figure_suite.py](../core/final_projection_figure_suite.py).

- SCAGR25 uses 24 cost directions. STAIR uses 10 coordinate directions.
- STAIR selects coordinates that can vary and have finite upper bounds on the original feasible set, using coordinate minimization and maximization LPs.
- Before clipping, the standard deviation of factor `j` is `0.70 * R_i * 0.82**(j - 1)`, where `R_i` is the sampling radius and `j` starts at one.
- A perturbation outside the ball of radius `R_i` is scaled back to its boundary.

Archived results use an earlier sampling configuration. When invoking a runner directly, choose a fresh output directory or pass its `--force` option to regenerate these cases. The root `reproduce_figures.py --rerun` command already passes `--force` to its runners.

Run the sampler checks from the repository root:

```bash
python -m unittest discover -s code/tests -p 'test_netlib_sampling.py' -v
```

## Archived plot metrics

For the LP plots, objective ratio is the returned objective divided by the Full objective. The code expresses these LPs as maximization problems. The outside-mass plot uses the saved standard errors for its bands. The training-sample plot uses sample standard deviations across held-out LPs. These plots are regenerated outputs of the archived LP package.
