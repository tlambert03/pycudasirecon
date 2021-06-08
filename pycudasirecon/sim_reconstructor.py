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
    """Main class for SIM reconstruction."""

    def __init__(self, image: np.ndarray, config: str = "") -> None:
        nz, ny, nx = image.shape
        self.image = image
        self.obj = SR_new(nx, ny, nz, config.encode())
        self.set_raw(image)
        self.get_result()

    def set_raw(self, array: np.ndarray) -> None:
        nz, ny, nx = array.shape
        SR_setRaw(self.obj, array, nx, ny, nz)

    def get_result(self):
        *_, ny, nx = self.image.shape
        nz = 9
        _result = np.empty((nz, ny * 2, nx * 2), dtype=np.float32)
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
    print(sr.get_recon_params())