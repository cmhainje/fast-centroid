# centroid

## Authors

- **Connor Hainje**, NYU

## Installation

```bash
pip install git+https://github.com/cmhainje/centroid
```

## usage

### off-the-shelf

if you have your own PSF, smooth the image with it and pass both the image and its smoothed counterpart to `centroid.detect_peaks()`. if you don't have a PSF, pass only the image and we'll smooth it with a unit-width Gaussian.

```python
centroid.detect_peaks(image, smooth, n_sigma=6, thresh=8)
# or
centroid.detect_peaks(image, n_sigma=6, thresh=8)
```

there are two knobs you can turn when using this function:

1. `n_sigma`: we compute an estimate of the noise level of `image` and only consider regions of the image where `smooth` exceeds `n_sigma` times this noise level. this can be disabled by setting `n_sigma=0`
2. `thresh`: we only consider regions where the brightnesses of pixels in `smooth` exceed this threshold. this can be disabled by setting `thresh=0`.

### fiddlier use cases

in case you're feeling fiddlier, here are the methods to know about:

- `centroid(image, peak_mask)`: this takes an image (probably want to use the PSF-smoothed image) and a `peak_mask`, which is a boolean array of the same shape as the image where elements are `True` if the element is a local maximum and `False` otherwise. then, `centroid` computes the centroid coordinate for each of the peaks in `peak_mask` using the polynomial fit described in [Vakili & Hogg 2016](https://arxiv.org/pdf/1610.05873).
- `identify_local_maxima(image, size=3)`: this takes an image and identifies local maxima. the result can be used as a `peak_mask` for `centroid`. by default, it looks for pixels which are the maximum of their local 3-by-3 neighborhood, but this can be changed with the `size` parameter (must be odd).
- `estimate_noise` estimates the noise level of a (hopefully background-subtracted) image using the algorithm from the [astrometry.net source](https://github.com/dstndstn/astrometry.net/blob/main/util/dsigma.inc).
- `identify_significant_regions` uses this noise estimate (or you can supply your own) to identify regions of pixels in the image with values greater than some number of sigma

for example, disabling all conditions on the peaks (i.e. noise level or minimum value) and considering only pixels which are the maximum of their local 5-by-5, we could centroid the sources as

```python
peak_mask = centroid.identify_local_maxima(image, size=5)
peaks = centroid.centroid(image, peak_mask)
```
