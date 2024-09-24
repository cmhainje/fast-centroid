"""
gaussians test

generate a 20x20 image with 5 Gaussian peaks at random locations. then, we use
`detect_peaks` to locate the peaks.
"""

import jax
import jax.numpy as jnp
import numpy as np

import centroid


# *** IMAGE GENERATION ***

# generate the peak locations
rng = np.random.default_rng(seed=0)
means = rng.uniform(1, 18, size=(5, 2))
means = jnp.array(sorted(means, key=lambda x: (x[0], x[1])))
dpsf = 0.5

# make the image
coords = jnp.stack(jnp.meshgrid(jnp.arange(20), jnp.arange(20), indexing='ij'), axis=-1)
# ^^ (row, col) coords like [[0, 0], [0, 1], [0, 2], ..., [19, 17], [19, 18], [19, 19]]
X = jnp.zeros((20, 20))
for mu in means:
    X += jax.scipy.stats.multivariate_normal.pdf(coords, mean=mu, cov=dpsf * jnp.eye(2))


# *** PEAK DETECTION ***

peak_mask = centroid.identify_local_maxima(X, size=3)
cs1 = centroid.centroid(X, peak_mask)
cs1 = jnp.array(sorted(cs1, key=lambda x: (x[0], x[1])))

cs2 = centroid.detect_peaks(X, n_sigma=0, thresh=0, dpsf=dpsf)
cs2 = jnp.array(sorted(cs2, key=lambda x: (x[0], x[1])))

dists1 = jnp.linalg.norm(means - cs1, axis=1)
dists2 = jnp.linalg.norm(means - cs2, axis=1)

for i, (p, c1, d1, c2, d2) in enumerate(zip(means, cs1, dists1, cs2, dists2)):
    print(f"Peak {i}:")
    print(f"  True:                       {p}")
    print(f"  Centroid on raw:            {c1}")
    print(f"  (distance):                 {d1:.3f}")
    print(f"  Centroid on matched filter: {c2}")
    print(f"  (distance):                 {d2:.3f}")
