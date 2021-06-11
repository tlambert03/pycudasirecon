try:
    import cupy as xp
except ImportError:
    print("could not import cupy, falling back to numpy & cpu")
    import numpy as xp


def forward_diff(data, dim):
    return xp.diff(data, axis=dim, append=0)


def back_diff(data, dim):
    return xp.diff(data, axis=dim, prepend=0)


def _fft_of_diff(sizex, sigma, mu, lamda):

    # FFTs of difference operator
    _v = xp.array([1, -2, 1])
    tmp_fft = xp.fft.fftn(_v[xp.newaxis, xp.newaxis, :], sizex)
    tmp_fft *= xp.conj(tmp_fft)
    divide = tmp_fft

    tmp_fft = xp.fft.fftn(_v[xp.newaxis, :, xp.newaxis], sizex)
    tmp_fft *= xp.conj(tmp_fft)
    divide += tmp_fft

    tmp_fft = xp.fft.fftn(_v[:, xp.newaxis, xp.newaxis], sizex)
    tmp_fft *= xp.conj(tmp_fft)
    divide += (sigma ** 2) * tmp_fft

    _v = xp.array([[1, -1], [-1, 1]])
    tmp_fft = xp.fft.fftn(_v[xp.newaxis, :, :], sizex)
    tmp_fft *= xp.conj(tmp_fft)
    divide += 2 * tmp_fft

    tmp_fft = xp.fft.fftn(_v[:, xp.newaxis, :], sizex)
    tmp_fft *= xp.conj(tmp_fft)
    divide += 2 * sigma * tmp_fft

    tmp_fft = xp.fft.fftn(_v[:, :, xp.newaxis], sizex)
    tmp_fft *= xp.conj(tmp_fft)
    divide += 2 * sigma * tmp_fft
    divide = (divide.real + mu / lamda).astype(xp.float32)

    return divide


def hessian(initial, iters=50, sigma=1, mu=150, lamda=1):
    initial = xp.asarray(initial, dtype="float32")

    if initial.ndim == 2 or initial.shape[0] < 3:
        sigma = 0
        print(
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

    ymax = initial.max()
    initial = initial / ymax
    sizex = initial.shape

    divide = _fft_of_diff(sizex, sigma, mu, lamda)

    # # #############
    # b1 = xp.zeros(sizex, "float32")
    # b2 = xp.zeros(sizex, "float32")
    # b3 = xp.zeros(sizex, "float32")
    # b4 = xp.zeros(sizex, "float32")
    # b5 = xp.zeros(sizex, "float32")
    # b6 = xp.zeros(sizex, "float32")

    bs = [xp.zeros(sizex, "float32") for _ in range(6)]

    x = xp.zeros(sizex, "int32")
    frac = (mu / lamda) * initial

    pairs = ((1, 1), (2, 2), (0, 0), (1, 2), (1, 0), (2, 0))

    for ii in range(iters):
        frac = xp.fft.fftn(frac)
        print(ii)
        if ii >= 1:
            x = xp.real(xp.fft.ifftn(frac / divide))
        else:
            x = xp.real(xp.fft.ifftn(frac / (mu / lamda)))

        frac = (mu / lamda) * initial

        for b, (i0, i1) in zip(bs, pairs):
            frac += _x(b, i0, i1, x, sigma, lamda)

        # u = back_diff(forward_diff(x, 1), 1)
        # signd = abs(u + b1) - 1 / lamda
        # signd[signd < 0] = 0
        # signd *= xp.sign(u + b1)
        # b1 += u - signd
        # frac += back_diff(forward_diff(signd - b1, 1), 1)

        # u = back_diff(forward_diff(x, 2), 2)
        # signd = abs(u + b2) - 1 / lamda
        # signd[signd < 0] = 0
        # signd *= xp.sign(u + b2)
        # b2 += u - signd
        # frac += back_diff(forward_diff(signd - b2, 2), 2)

        # u = back_diff(forward_diff(x, 0), 0)
        # signd = abs(u + b3) - 1 / lamda
        # signd[signd < 0] = 0
        # signd *= xp.sign(u + b3)
        # b3 += u - signd
        # frac += (sigma ** 2) * back_diff(forward_diff(signd - b3, 0), 0)

        # u = forward_diff(forward_diff(x, 1), 2)
        # signd = abs(u + b4) - 1 / lamda
        # signd[signd < 0] = 0
        # signd *= xp.sign(u + b4)
        # b4 += u - signd
        # frac += 2 * back_diff(back_diff(signd - b4, 2), 1)

        # u = forward_diff(forward_diff(x, 1), 0)
        # signd = abs(u + b5) - 1 / lamda
        # signd[signd < 0] = 0
        # signd *= xp.sign(u + b5)
        # b5 += u - signd
        # frac += 2 * sigma * back_diff(back_diff(signd - b5, 0), 1)

        # u = forward_diff(forward_diff(x, 2), 0)
        # signd = abs(u + b6) - 1 / lamda
        # signd[signd < 0] = 0
        # signd *= xp.sign(u + b6)
        # b6 += u - signd
        # frac += 2 * sigma * back_diff(back_diff(signd - b6, 0), 2)

    x[x < 0] = 0
    if x.shape != initial.shape:
        x = x[: initial.shape[0], :, :]
    return x * ymax


def _x(b, i0, i1, x, sigma, lamda):
    u = forward_diff(forward_diff(x, i0), i1)
    signd = abs(u + b) - 1 / lamda
    signd[signd < 0] = 0
    signd *= xp.sign(u + b)
    b += u - signd
    return 2 * sigma * back_diff(back_diff(signd - b, i1), i0)

if __name__ == '__main__':
    import tifffile
    
    print("read")
    im = tifffile.imread('/Users/talley/Desktop/SIMrecon_test_data/5phases-need-deskew/GPUsirecon/sample_scan_488_8ms_zp179um_SIM_5Phase_proc.tif')
    print("start")
    print(hessian(im).shape)
