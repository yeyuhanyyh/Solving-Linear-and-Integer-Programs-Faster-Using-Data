# Benchmark data provenance

The following files are expanded MPS versions of the GROW7 and SC205 linear
programming benchmarks distributed by [Netlib's LP data collection](https://netlib.org/lp/data/):

| Bundled file | MPS `NAME` header | Upstream compressed MPS |
|---|---|---|
| `data/netlib/selected/grow7.mps` | `GROW7` | [grow7](https://netlib.org/lp/data/grow7) |
| `data/netlib/selected/sc205.mps` | `SC205` | [sc205](https://netlib.org/lp/data/sc205) |

The [Netlib LP data README](https://netlib.org/lp/data/readme) describes the
compressed format, expansion to standard MPS, benchmark dimensions, and data
contributors. It credits Bob Fourer with supplying both GROW7 and SC205.

These benchmark inputs are third-party data and are excluded from this
repository's BSD 3-Clause software license, as stated in `LICENSE`. This notice
records their source and does not grant additional rights or assign a license
to Netlib's data. Consult the upstream collection for its documentation and
applicable terms.

The current UC input in `ip/current/data/` derives from the public
`matpower/case14/2017-02-01` benchmark in
[UnitCommitment.jl v0.4.3](https://github.com/ANL-CEEESA/UnitCommitment.jl/tree/v0.4.3).
Physical parameters and network data are retained. Generator costs use the
ranges of the [MIPLearn v0.4 UC generator](https://anl-ceeesa.github.io/MIPLearn/0.4/guide/problems/#unit-commitment).
The arrays contain an exactly equivalent formulation obtained by eliminating
redundant auxiliary variables. This notice does not assign a new license to
upstream data.

Grouped and correlated knapsack inputs are synthetic. The latter uses the
almost-strongly-correlated coefficient distribution studied by Pisinger (2005),
adapted to fixed weights and varying profits. The older `ip/snapshot/` UC and
knapsack inputs are synthetic and remain a historical experiment archive.
