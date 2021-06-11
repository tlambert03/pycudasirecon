import warnings

import numpy as np

try:
    import cupy

except ImportError:
    cupy = None

xp = np if cupy is None else cupy


def hessian_denoise(initial: np.ndarray, iters=50, mu=150, sigma: float = 1, lamda=1):
    """Hessian (Split-Bregman) SIM denoise algorithm.

    May be applied to images images reconstructed with Wiener method to further
    reduce noise.

    Adapted from MATLAB code published in Huang et al 2018 [1]_.
    See supplementary information (fig 7) for details on the paramaeters.

    Parameters
    ----------
    initial : np.ndarray
        input image
    iters : int, optional
        number of iterations, by default 50
    mu : int, optional
        tradeoff between resolution and denoising. Higher values result in increased
        resolution but increased noise, by default 150
    sigma : float, optional
        When structures change slowly along the Z/T axis, use a value between
        0 and 1 to obtain trade-off between effective denoising and minimal temporal
        blurring. Set to zero to turn off regularization along the Z/T axis.
        by default 1
    lamda : int, optional
        [description], by default 1

    Returns
    -------
    result: np.ndarray
        denoised image

    References
    ----------

    .. [1] Huang, X., Fan, J., Li, L. et al. Fast, long-term, super-resolution imaging
           with Hessian structured illumination microscopy. Nat Biotechnol 36, 451–459
           (2018). https://doi-org.ezp-prod1.hul.harvard.edu/10.1038/nbt.4115

    """
    if xp is not cupy:
        warnings.warn("could not import cupy... falling back to numpy & cpu.")

    initial = xp.asarray(initial, dtype="single")

    if initial.ndim == 2 or initial.shape[0] < 3:
        sigma = 0
        warnings.warn(
            "Number of Z/T planes is smaller than 3, the t and z-axis of "
            "Hessian was turned off(sigma=0)"
        )
        if initial.ndim == 2:
            initial = xp.tile(initial, [3, 1, 1])
        elif initial.ndim == 3:
            if initial.shape[0] == 1:
                initial = xp.tile(initial[-1], [3, 1, 1])
            elif initial.shape[0] == 2:
                initial = xp.concatenate([initial, xp.expand_dims(initial[-1], 0)], 0)

    _mu_d_lamda = mu / lamda
    ymax = initial.max()
    initial /= ymax
    sizex = initial.shape

    divide = (_fft_of_diff(sizex, sigma) + _mu_d_lamda).astype(xp.float32)

    bs = xp.zeros((6,) + sizex, "float32")
    x = xp.zeros(sizex, "int32")
    frac = _mu_d_lamda * initial

    for ii in range(iters):
        frac = xp.fft.fftn(frac)

        divisor = divide if ii >= 1 else _mu_d_lamda
        x = xp.fft.ifftn(frac / divisor).real

        frac = _mu_d_lamda * initial
        frac += _bfbf(bs[0], 1, 1, x, lamda)
        frac += _bfbf(bs[1], 2, 2, x, lamda)
        frac += _bfbf(bs[2], 0, 0, x, lamda)
        frac += _ffbb(bs[3], 1, 2, x, lamda)
        frac += sigma * _ffbb(bs[4], 1, 0, x, lamda)
        frac += sigma * _ffbb(bs[5], 2, 0, x, lamda)

    x.clip(0, out=x)
    if x.shape != initial.shape:
        x = x[: initial.shape[0]]
    x *= ymax

    return x.get() if hasattr(x, "get") else x


def _fft_of_diff(shape, sigma):
    # FFTs of difference operator
    _v0 = xp.array([1, -2, 1])
    _v1 = xp.array([[1, -1], [-1, 1]])
    divide = (
        _fconj(_v0.reshape((1, 1, 3)), shape)
        + _fconj(_v0.reshape((1, 3, 1)), shape)
        + (sigma ** 2) * _fconj(_v0.reshape((3, 1, 1)), shape)
        + 2 * _fconj(_v1.reshape((1, 2, 2)), shape)
        + 2 * sigma * _fconj(_v1.reshape((2, 1, 2)), shape)
        + 2 * sigma * _fconj(_v1.reshape((2, 2, 1)), shape)
    )
    return divide.real


def _fconj(array, shape):
    tmp_fft = xp.fft.fftn(array, shape)
    return tmp_fft * xp.conj(tmp_fft)


def _forward_diff(data, axis):
    return xp.diff(data, axis=axis, append=0)


def _back_diff(data, axis):
    return xp.diff(data, axis=axis, prepend=0)


def _middle(u, b_, lamda):
    signd = abs(u + b_) - 1 / lamda
    signd.clip(0, out=signd)
    signd *= xp.sign(u + b_)
    b_ += u - signd
    return signd


def _bfbf(b_, c0, c1, x, lamda):
    u = _back_diff(_forward_diff(x, c0), c1)
    signd = _middle(u, b_, lamda)
    return _back_diff(_forward_diff(signd - b_, c1), c0)


def _ffbb(b_, c0, c1, x, lamda):
    u = _forward_diff(_forward_diff(x, c0), c1)
    signd = _middle(u, b_, lamda)
    return 2 * _back_diff(_back_diff(signd - b_, c1), c0)
