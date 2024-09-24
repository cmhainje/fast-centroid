import jax
import jax.numpy as jnp


def estimate_noise(image, sp=5, gridsize=20):
    """
    Parameters:
        image (Array): image
        sp (int): spacing between points in each pair
        gridsize (int): size of grid for estimates
    """
    from jax.scipy.special import erf

    # JAXifying `dsigma`
    nr, nc = image.shape
    dr = max(min(gridsize, nr // 4), 1)
    dc = max(min(gridsize, nc // 4), 1)

    diff = jnp.abs(image[:-sp:dr, :-sp:dc] - image[sp::dr, sp::dc]).flatten()
    diff = jnp.sort(diff)
    ndiff = len(diff)

    s, Nsigma = 0.0, 0.7
    ROOT_2 = jnp.sqrt(2)
    while s == 0.0:
        k = jnp.floor(ndiff * erf(Nsigma / ROOT_2)).astype(int)

        if k >= ndiff:
            print('warning: noise estimate failed')
            s = 0.0
            break

        s = diff[k] / (Nsigma * ROOT_2)
        Nsigma += 0.1

    return s


def identify_signifcant_regions(image, sigma=None, n_sigma=6.0, dpsf=1.0):
    """
    Parameters:
        image (Array): image
        sigma (float | None): noise level, defaults to None.
            If None, calls `estimate_noise` with default parameters.
        plim (float): number of sigma to count as significant
    """
    if sigma is None:
        sigma = estimate_noise(image)

    limit = (sigma / (2.0 * jnp.sqrt(jnp.pi) * dpsf)) * n_sigma
    return image >= limit
