# AI Olymp 2026 Qualifiers (Kazakhstan) — Write-up

[Solutions code is here](https://github.com/abukanabek/ai-solutions/tree/main/competitions/kazakhstan-ai-olympiad/2026/qualifiers)

**Warning** — some of them do not match the solutions described here!

## A. YAMAZAKI OPTIMIZER

TODO

(short idea: brute-force over optimizers and find the one that best matches the given logs)

## B. NIPS SUBMISSION

### Solution

This task can be solved with a simple fit + predict. Train a classifier on the labeled data, then predict on the test data. A basic `LogisticRegression` gets full score.

## C. CORGI CLASSIFICATION

### Solution

The key difference between the two breeds is the value of `X²+Y²`. In particular, for pembrokes it is usually larger (you could notice this either by writing generator functions for the two classes or just by intuition). So for each population we compute the average `X²+Y²` over all dogs in that population, getting 100 values total (one per population). We classify populations with this value below the median as corgis, and those above as pembrokes. This solution gets full score.

## Warning

Not all competitive-programming tasks got full score in `python` (TLE). So it’s recommended to implement solutions in `C++`.

## D. Iterated Sigmoid

### Solution

Notice that if you apply the sigmoid operation many times, the changes become smaller and smaller. The problem only requires precision up to 6 digits after the decimal point, so if the change is already less than `1e-6` (for safety you can use `1e-7` or smaller), we stop and print the answer.

Another method is to do `n = min(30, n)`. This also limits how many times we apply the operation.

Complexity: `O(T)` (second method)

## E. Attention Variance

### Solution

This task can be solved with prefix sums. Instead of recomputing what each query asks for every time, we precompute values once before processing queries.

Let `pref` and `pref_sq` be arrays such that `pref[i]` stores the sum of the first `i` numbers in the array, and `pref_sq[i]` stores the sum of squares of the first `i` numbers.

Then the answer for a query `(l, r)` is:

`k * (pref_sq[r] - pref_sq[l-1]) - (pref[r] - pref[l-1])²`

Complexity: `O(n + q)` (each query is `O(1)`)

## F. Capped Attention

### Solution

This task can be solved using binary search and prefix sums.

Sort tokens by `k`.

Then for a fixed `i`, the contribution from `j` is:

- if `q[i] * k[j] < T`, add `q[i] * k[j] * v[j]`
- otherwise add `T * v[j]`

After sorting by `k`, the saturation boundary is the first index `edge` where `q[i] * k[edge] >= T`. We find it with binary search.

To compute sums fast, precompute:

- `pref[i] = sum_{0..i} (k * v)`
- `suf[i] = sum_{i..n-1} v`

Then:

- if `edge == -1` (no saturation at all), `C[i] = q[i] * pref[n-1]`
- otherwise `C[i] = q[i] * pref[edge-1] + T * suf[edge]` (if `edge-1 < 0`, the first part is 0)

Store answers back by original `ind` (because we sorted).

Complexity: `O(n log n)` due to sorting and `n` binary searches.
