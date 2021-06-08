from typing import Tuple, Union
import numpy as np

from ._libwrap import (
    ImageParams,
    ReconParams,
    SR_getImageParams,
    SR_getReconParams,
    SR_getResult,
    SR_new,
    SR_setRaw,
)


class SIMReconstructor:
    """Main class for SIM reconstruction.


    Parameters
    ----------
    arg0 : Union[np.ndarray, Tuple[int, int, int]]
        Either a numpy array of raw data, or a shape tuple (3 integers) that indicate
        the size of raw data (to be provided later with `set_raw`).
    config : str, optional
        config file path, by default ""

    Raises
    ------
    ValueError
        If the array is not ndim==3 or shape is not a 3-tuple
    """

    def __init__(
        self, arg0: Union[np.ndarray, Tuple[int, int, int]], config: str = ""
    ) -> None:

        if isinstance(arg0, np.ndarray):
            if not arg0.ndim == 3:
                raise ValueError("array must have 3 dimensions")
            image = arg0
            self.shape = image.shape
        elif isinstance(arg0, (list, tuple)):
            if not len(arg0) == 3:
                raise ValueError("shape argument must have length 3")
            image = None
            self.shape = arg0
        nz, ny, nx = self.shape
        self.obj = SR_new(nx, ny, nz, config.encode())
        if image is not None:
            self.set_raw(image)

    def set_raw(self, array: np.ndarray) -> None:
        nz, ny, nx = array.shape
        SR_setRaw(self.obj, array, nx, ny, nz)

    def get_result(self):
        *_, ny, nx = self.shape
        rp = self.get_recon_params()
        nz = int(self.get_image_params().nz * rp.z_zoom)
        out_shape = (nz, int(ny * rp.zoomfact), int(nx * rp.zoomfact))
        _result = np.empty(out_shape, np.float32)
        SR_getResult(self.obj, _result)
        return _result

    def get_recon_params(self) -> ReconParams:
        return ReconParams.from_address(SR_getReconParams(self.obj))

    def get_image_params(self) -> ImageParams:
        return ImageParams.from_address(SR_getImageParams(self.obj))


if __name__ == "__main__":
    from pathlib import Path

    import tifffile as tf

    BASE = Path("/home/tjl10/dev/cudasirecon")
    CONFIG = str(BASE / "test_data/config-tiff")

    img = tf.imread(str(BASE / "test_data/raw.tif"))
    print("img: ", img.shape, img.dtype)
    sr = SIMReconstructor(img, CONFIG)
    sr.get_result()
    print(sr.get_recon_params())
