import jax
import jax.numpy as jnp

from .noise import estimate_noise, identify_signifcant_regions
from .smooth import gauss_smooth
from .centroid import centroid


def identify_local_maxima(image, size: int = 3):
    """
    Parameters:
        image (Array): image
        size (int): size of the neighborhood to check for maximum.
            must be odd. defaults to 3.
    """
    if size <= 0:
        raise ValueError('size must be positive')
    if size % 2 != 1:
        raise ValueError('size must be odd')

    half = size // 2

    # pad image by half size on all sides
    pad_image = jnp.pad(image, half, constant_values=jnp.nan)

    def is_maximum(i, j):
        center = jax.lax.dynamic_slice(
            pad_image, [i + half, j + half], [1, 1]
        ).squeeze()
        neighborhood = jax.lax.dynamic_slice(
            pad_image, [i, j], [size, size]
        )
        return center == neighborhood.max()

    v_is_maximum = jax.vmap(is_maximum, in_axes=(None, 0), out_axes=0)
    vv_is_maximum = jax.vmap(v_is_maximum, in_axes=(0, None), out_axes=0)

    nr, nc = image.shape
    return vv_is_maximum(jnp.arange(nr), jnp.arange(nc))


def detect_peaks(image, smooth=None, dpsf: float=1.0, n_sigma: float=6, thresh: float=8):
    """
    todo: add background subtraction

    Parameters:
        image (Array): image
        smooth (Array, optional): smoothed image. if None, the smoothed image will
            be computed using a Gaussian filter of width `dpsf`.
        dpsf (float): width of the Gaussian filter to use for smoothing if `smooth`
            is None, defaults to 1.0
        n_sigma (float): number of sigma to count as significant, defaults to 6
        thresh (float): absolute minimum peak value, defaults to 8
    """
    if smooth is None:
        smooth = gauss_smooth(image, sigma=dpsf, size=None)

    mask = smooth > thresh
    mask &= identify_local_maxima(smooth)
    if n_sigma > 0:
        mask &= identify_signifcant_regions(
            smooth, sigma=estimate_noise(image), n_sigma=n_sigma
        )
    return centroid(smooth, mask)
