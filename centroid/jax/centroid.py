import jax
import jax.numpy as jnp

_d = jnp.array([-1, 0, +1])
_r = jnp.tile(_d[:, None], (1, 3)).flatten()
_c = jnp.tile(_d[None, :], (3, 1)).flatten()

# decision matrix
A = jnp.stack([
    jnp.ones_like(_r),
    _r,
    _c,
    _r ** 2,
    _r * _c,
    _c **2,
], axis=1)


def centroid(image, peak_mask, reg=1e-5, disable_checks=False):
    """
    Given an image and a mask indicating local maxima, performs a
    parabolic fit to estimate the centroid of each peak.

    Parameters:
        image (Array): image
        peak_mask (Array<bool>): array of same shape as image.
            elements are True for local maxima, False otherwise.
        reg (float): regularization parameter, defaults to 1e-4.
        disable_checks (bool): by default (False), the method checks that the
            peaks are all finite, within the image, and offset by no more
            than 1 along either axis from the original peak coordinates.
            if any of these conditions are failed, the returned centroid
            for that peak is NaN. if this argument is True, these checks
            are disabled.

    Returns:
        Array: array of shape (n_peaks, 2) containing the centroid coordinates,
        where `n_peaks` is the number of True elements in `peak_mask`.
    """

    pad_image = jnp.pad(image, 1)

    def single_peak(r, c):
        def _lstsq(a, b):
            return jnp.linalg.lstsq(a, b, rcond=None)[0].squeeze()

        z = jax.lax.dynamic_slice(pad_image, [r, c], [3, 3]).reshape((-1, 1))
        reg_matrix = jnp.var(z) * jnp.eye(2)

        a, b, c, d, e, f = _lstsq(A.T @ A, A.T @ z)
        D = jnp.array([[2 * d, e], [e, 2 * f]]) + reg * reg_matrix
        r_c, c_c = _lstsq(D, jnp.array([[-b, -c]]).T)
        return jnp.array([r_c, c_c])

    v_peak = jax.vmap(single_peak, in_axes=(0, 0), out_axes=0)
    peaks_0 = jnp.argwhere(peak_mask)
    rows, cols = peaks_0.T
    d_peaks = v_peak(rows, cols)
    peaks = peaks_0 + d_peaks

    # ensure finite, within 3x3, and within image
    finite = jnp.isfinite(peaks).all(axis=1)
    within_3x3 = (jnp.abs(d_peaks) <= 1).all(axis=1)
    within_img = (
        (peaks_0[:, 0] > 0) &
        (peaks_0[:, 0] < image.shape[0] - 1) &
        (peaks_0[:, 1] > 0) &
        (peaks_0[:, 1] < image.shape[1] - 1)
    )
    mask = finite & within_3x3 & within_img

    peaks = peaks.at[~mask].set(jnp.nan)
    return peaks
