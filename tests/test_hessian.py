from pathlib import Path

import numpy as np
import tifffile

from pycudasirecon import hessian_denoise

DATA = Path(__file__).parent / "data"
WEINER_RESULT = DATA / "expected.tif"
HESS_RESULT = DATA / "hess9.tif"


def test_hessian():
    im = tifffile.imread(WEINER_RESULT)
    out = hessian_denoise(im)
    assert np.allclose(out[9], tifffile.imread(HESS_RESULT))
