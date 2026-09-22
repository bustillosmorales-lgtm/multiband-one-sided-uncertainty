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
| `code/fibre_orthogonality.py` | The orthogonality of the fibre matrix `W(zeta)` (Lemma 2.2) for 2, 3 and 4 components, with the normalizations of Casini–Huerta and of Hollands. |
| `code/leak_setup.py` | Shared setup: the set `(0,0.4) ∪ (0.6,1)`, the quadrature, the transform `U` and the quantities of each test. |
| `code/leak_moment_tests.py` | The leak identity (Lemma 2.3) and the moment bound (Proposition 3.1) on 16 bump test functions. |
| `code/packets.py` | The packets of the sharpness proof (Lemma 4.1), computed in the variable `zeta`: leak against the exact Gaussian value, first moment, the approach to `kappa(K)` as the width decreases, and the effect of removing the companion amplitude. |
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

## Licence

MIT, see `LICENSE`.
