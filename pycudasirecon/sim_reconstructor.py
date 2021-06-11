import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple, Union

import numpy as np

from ._libwrap import ImageParams as SR_ImageParams
from ._libwrap import ReconParams as SR_ReconParams
from ._libwrap import (
    SR_getImageParams,
    SR_getReconParams,
    SR_getResult,
    SR_loadAndRescaleImage,
    SR_new_from_shape,
    SR_processOneVolume,
    SR_setCurTimeIdx,
    SR_setRaw,
)
from ._recon_params import ReconParams
from ._util import caplog


@contextmanager
def temp_config(**kwargs):
    params = ReconParams(**kwargs)
    tf = NamedTemporaryFile(delete=False)
    tf.file.write(params.to_config().encode())  # type: ignore
    tf.close()
    try:
        yield tf
    finally:
        os.unlink(tf.name)


def reconstruct(
    array, psf: Optional[np.ndarray] = None, otf_file: str = None, **kwargs
) -> np.ndarray:
    with temp_config(otf_file=otf_file, **kwargs) as cfg:
        return SIMReconstructor(array, cfg.name).get_result()


class SIMReconstructor:
    """Main class for SIM reconstruction.

    Parameters
    ----------
    arg0 : Union[np.ndarray, Tuple[int, int, int]]
        Either a numpy array of raw data, or a shape tuple (3 integers) that indicate
        the size of raw data (to be provided later with `set_raw`).
    config : str, optional
        Config file path (overrides kwargs), by default None
    **kwargs
        valid ReconParams Fields


    Raises
    ------
    ValueError
        If the array is not ndim==3 or shape is not a 3-tuple
    """

    def __init__(
        self,
        arg0: Union[np.ndarray, Tuple[int, int, int]],
        config: str,
    ) -> None:
        image: Optional[np.ndarray]
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
        with caplog():
            self._ptr = SR_new_from_shape(nx, ny, nz, config.encode())

        if image is not None:
            self.set_raw(image)
            self.process_volume()

    def process_volume(self):
        with caplog():
            SR_processOneVolume(self._ptr)

    def set_raw(self, img: np.ndarray) -> None:
        nz, ny, nx = img.shape
        if not np.issubdtype(img.dtype, np.float32) or not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img, dtype=np.float32)
        with caplog():
            SR_setRaw(self._ptr, img, nx, ny, nz)
            SR_loadAndRescaleImage(self._ptr, 0, 0)
            SR_setCurTimeIdx(self._ptr, 0)

    def get_result(self) -> np.ndarray:
        *_, ny, nx = self.shape
        rp = self.get_recon_params()
        nz = int(self.get_image_params().nz * rp.z_zoom)
        out_shape = (nz, int(ny * rp.zoomfact), int(nx * rp.zoomfact))
        _result = np.empty(out_shape, np.float32)
        SR_getResult(self._ptr, _result)
        return _result

    def get_recon_params(self) -> SR_ReconParams:
        return SR_ReconParams.from_address(SR_getReconParams(self._ptr))

    def get_image_params(self) -> SR_ImageParams:
        return SR_ImageParams.from_address(SR_getImageParams(self._ptr))
