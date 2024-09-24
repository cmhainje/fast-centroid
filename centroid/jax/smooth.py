import jax
import jax.numpy as jnp
from typing import Optional


def make_gaussian_filter(sigma=1.0, size=None, normalize_sum=False):
    """
    Parameters:
        sigma (float): variance of the Gaussian
        size (int | None): must be odd, defaults to None (estimated from sigma)
        normalize_sum (bool): whether to normalize the sum of the filter
            values to add to 1, defaults to False
    """

    if size is not None:
        if size <= 0:
            raise ValueError("size must be positive")
        elif size % 2 != 1:
            raise ValueError("size must be odd")

    else:
        size = 2 * int(jnp.ceil(2 * sigma)) + 1

    middle = size // 2

    a = jnp.arange(size)
    dsq = (a[:, None] - middle)**2 + (a[None, :] - middle)**2
    G = jnp.exp(-dsq / (2 * sigma**2))

    if normalize_sum:
        G /= G.sum()
    else:
        G /= (2 * jnp.pi * sigma**2)

    return G


def gauss_smooth(image, sigma: float=1, size: Optional[int]=3):
    """
    Parameters:
        image (Array): image
        sigma (float): width of the Gaussian kernel, defaults to 1.0
        size (int | None): size of the kernel, defaults to 3.
            If None, the necessary size is estimated from the given sigma.
    """
    from jax.scipy.signal import convolve2d

    g = make_gaussian_filter(sigma=sigma, size=size)
    return convolve2d(image, g, mode='same')
