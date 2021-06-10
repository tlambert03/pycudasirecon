from typing import Tuple, Union

import numpy as np

from ._libwrap import (
    ImageParams,
    ReconParams,
    SR_getImageParams,
    SR_getReconParams,
    SR_getResult,
    SR_loadAndRescaleImage,
    SR_new_from_shape,
    SR_setCurTimeIdx,
    SR_setRaw,
    SR_processOneVolume,
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
        self._ptr = SR_new_from_shape(nx, ny, nz, config.encode())

        if image is not None:
            self.set_raw(image)
            self.process_volume()

    def process_volume(self):
        SR_processOneVolume(self._ptr)

    def set_raw(self, array: np.ndarray) -> None:
        nz, ny, nx = array.shape
        SR_setRaw(self._ptr, array, nx, ny, nz)
        SR_loadAndRescaleImage(self._ptr, 0, 0)
        SR_setCurTimeIdx(self._ptr, 0)

    def get_result(self):
        *_, ny, nx = self.shape
        rp = self.get_recon_params()
        nz = int(self.get_image_params().nz * rp.z_zoom)
        out_shape = (nz, int(ny * rp.zoomfact), int(nx * rp.zoomfact))
        _result = np.empty(out_shape, np.float32)
        SR_getResult(self._ptr, _result)
        return _result

    def get_recon_params(self) -> ReconParams:
        return ReconParams.from_address(SR_getReconParams(self._ptr))

    def get_image_params(self) -> ImageParams:
        return ImageParams.from_address(SR_getImageParams(self._ptr))


if __name__ == "__main__":
    from pathlib import Path

    import tifffile as tf

    ROOT = Path(__file__).parent.parent
    CONFIG = str(ROOT / "tests/data/config")

    img = tf.imread(str(ROOT / "tests/data/raw.tif"))
    print("img: ", img.shape, img.dtype)
    sr = SIMReconstructor(img, CONFIG)
    sr.get_result()
    print(sr.get_recon_params())
