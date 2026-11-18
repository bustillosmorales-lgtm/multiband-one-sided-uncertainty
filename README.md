# The sharp one-sided uncertainty principle for signals with spectrum in several intervals: code and data

Python code and computed results accompanying the article

> F. Bustillos, *The sharp one-sided uncertainty principle for signals with spectrum in several intervals*.

Let `K = (a_1,b_1) ∪ … ∪ (a_n,b_n)` be a finite union of disjoint intervals and let `F` be a signal with spectrum in `K`. The article proves that the energy of `F` on the negative half-line, `mu`, and its first moment on the positive half-line, `D`, satisfy

```
mu  >=  C ||phi||^2 exp(-pi^2 kappa D / ||phi||^2)      for every kappa > kappa(K),
```

and that no smaller exponent is possible, where

```
kappa(K) = 4 / min_{x in K} z'(x),      z'(x) = sum_i ( 1/(x - a_i) - 1/(x - b_i) ).
```

For one interval `kappa(K)` is its length. For two or more intervals it lies strictly above the length of every component and at most the length of the convex hull, and for `K_r = (-1,-r) ∪ (r,1)` it equals `2 (1 - sqrt r)(1 + r)/(1 + sqrt r)`. The proof uses the diagonalization of the finite Hilbert transform on a union of intervals found by Casini and Huerta, for which the article gives a self-contained proof.

## Contents

| File | What it computes |
|---|---|
| `code/fibre_orthogonality.py` | The orthogonality of the fibre matrix `W(zeta)` (Lemma 3.2) for 2, 3 and 4 components, with the normalizations of Casini–Huerta and of Hollands. |
| `code/leak_setup.py` | Shared setup: the set `(0,0.4) ∪ (0.6,1)`, the quadrature, the transform `U` and the quantities of each test. |
| `code/leak_moment_tests.py` | The leak identity (Lemma 3.3) and the moment bound (Proposition 4.1) on 16 bump test functions. |
| `code/packets.py` | The packets of the sharpness proof (Lemma 5.1), computed in the variable `zeta`: leak against the exact Gaussian value, first moment, the approach to `kappa(K)` as the width decreases, and the effect of removing the companion amplitude. |
| `code/toeplitz_multiband.py` | Smallest eigenvalue of principal submatrices of the Toeplitz matrix with symbol `xi` on multiband index sets, `m = 2^8 … 2^16`. |
| `code/toeplitz_keff.py` | The effective constant `kappa_eff` of the table in the numerical section. |
| `code/toeplitz_second_order.py` | The second-order heuristic: the constants `c_K` of the packets and the growth of the Toeplitz deviation. |
| `code/verify_all.py` | Recomputes every numerical value quoted in the article and reports discrepancies. |
| `results/` | The outputs of the scripts above, as produced on the machine described in the article. |

## Reproducing

Run the scripts from the `code/` folder, since they read and write `../results/`:

```
python -m pip install numpy scipy
cd code
python verify_all.py                                        # 59 checks against ../results/
python fibre_orthogonality.py > ../results/fibre_orthogonality.txt
python leak_moment_tests.py   > ../results/leak_moment_tests.txt
python packets.py             > ../results/packets.txt
python toeplitz_multiband.py 16 > ../results/toeplitz_multiband.txt
python toeplitz_keff.py       > ../results/toeplitz_keff.txt
python toeplitz_second_order.py > ../results/toeplitz_second_order.txt
```

`verify_all.py` prints one line per quoted value, with the value in the article and the value recomputed, and ends with the number of failures. On the stored results it reports `59 checks, 0 FAIL`.

Runtimes on a desktop machine with 22 logical cores: seconds for `fibre_orthogonality.py` and `verify_all.py`, about 15 minutes for `leak_moment_tests.py`, 20 minutes for `packets.py`, 10 minutes for `toeplitz_multiband.py` and 5 minutes for `toeplitz_keff.py`.

## High-precision frontier (numerical section, Table of slopes)

The optimal frontier of Section 6 is computed in multiprecision arithmetic with `python-flint`.

| File | What it does |
|---|---|
| `code/alta_precision/hp.py` | Builds the matrices of the leak and of the one-sided moment in the frequency domain: the Hilbert transform of each basis function is obtained exactly as a polynomial times a logarithm plus a smooth part, in interval arithmetic with 1200 bits. The basis is `(1-s^2)^4` times the Jacobi polynomials on each component. |
| `code/alta_precision/run.py` | Minimizes `mu + lambda D` with `||phi|| = 1` by inverse iteration with continuation in `lambda`, with working precision raised adaptively from 500 bits, and writes one record per value of `lambda` to `par_<tag>.jsonl`. |
| `code/alta_precision/fit.py` | Reads the `par_*.jsonl` records, computes the local slope of `-log(mu_min)` against `D`, extrapolates it in the degree of the basis and fits `s(D) = s_inf - c D^(-p)`. |
| `results/alta_precision/par_*.jsonl` | The records of every run reported in the article. |
| `results/alta_precision/fit_output.txt` | The output of `fit.py` on those records. |

The sets are `0` (one interval of length 1, the control), `A` (K_A of the article) and `B` (K_B of the article), scaled to total width 1.
Usage of `run.py`: `python run.py <set> <degree> <time-grid> <lambda_0> <steps> <steps-per-decade> [<bits> <suffix>]`.
It writes in the working folder, and `fit.py` reads the `par_*.jsonl` of the working folder, so run both from `results/alta_precision/` with `PYTHONPATH=../../code/alta_precision`. The runs of the article were:

| Records | Command | Largest D | Hours |
|---|---|---|---|
| `par_0_d120_h128s.jsonl` | `python run.py 0 120 128 0.0001 50 3 1200 s` | 6.36 | 0.0 |
| `par_0_d120_h128x.jsonl` | `python run.py 0 120 128 0.0001 110 3 1200 x` | 11.70 | 0.1 |
| `par_0_d160_h128x.jsonl` | `python run.py 0 160 128 0.0001 110 3 1200 x` | 11.55 | 0.2 |
| `par_0_d50_h128.jsonl` | `python run.py 0 50 128 0.01 200 8` | 8.66 | 0.5 |
| `par_0_d80_h128.jsonl` | `python run.py 0 80 128 0.01 200 8` | 8.21 | 0.1 |
| `par_A_d120_h128s.jsonl` | `python run.py A 120 128 0.0001 50 3 1200 s` | 9.60 | 0.3 |
| `par_A_d160_h128x.jsonl` | `python run.py A 160 128 0.0001 70 3 1200 x` | 12.13 | 0.6 |
| `par_A_d50_h128.jsonl` | `python run.py A 50 128 0.01 200 8` | 13.10 | 0.1 |
| `par_A_d80_h128.jsonl` | `python run.py A 80 128 0.01 200 8` | 12.39 | 0.5 |
| `par_B_d100_h128s.jsonl` | `python run.py B 100 128 0.0001 50 3 1200 s` | 15.22 | 0.6 |
| `par_B_d50_h128.jsonl` | `python run.py B 50 128 0.01 200 8` | 20.84 | 0.7 |
| `par_B_d80_h128s.jsonl` | `python run.py B 80 128 0.0001 50 3 1200 s` | 15.33 | 0.4 |

`fit.py` prints the values of the table of slopes (`python ../../code/alta_precision/fit.py > fit_output.txt`). The stored `fit_output.txt` was regenerated from the stored records with the scripts of this folder.
Required packages: `python-flint`, `numpy`, `scipy`.

## Licence

MIT, see `LICENSE`.
