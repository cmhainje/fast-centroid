import jax
import jax.numpy as jnp

import centroid

X = jnp.zeros((10, 10))

peaks = [
    [1, 3],
    [5, 5],
    [8, 7]
]
for r, c in peaks:
    X = X.at[r, c].set(1)

print(centroid.detect_peaks(X, n_sigma=0, thresh=0))
