from __future__ import annotations

import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import numpy as np

from ._lib import lib
from ._otf import PathLike, temporary_otf
from ._recon_params import temp_config

# TODO: make sure that caplog works on ipython and windows
# before using it
caplog = nullcontext


def reconstruct(
    array: np.ndarray,
    otf: np.ndarray | PathLike | None = None,
    psf: np.ndarray | PathLike | None = None,
    makeotf_kwargs: dict | None = None,
    otf_file: PathLike | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Perform SIM reconstruction on array.

    Parameters
    ----------
    array : np.ndarray
        The array to reconstruct
    otf : array or str or Path, optional
        OTF to use for reconstruction. Either PSF or OTF must be provided,
        by default None
    psf : array or str or Path, optional
        PSF to use for reconstruction. Either PSF or OTF must be provided,
        by default None
    makeotf_kwargs : dict
        If `psf` is provided, these kwargs will be passed to the
        :func:`~pycudasirecon.make_otf` function when generating the OTF.
        See `make_otf` docstring for details.
    otf_file : str or Path, optional
        OTF file to use for reconstruction.  Either PSF or OTF must be provided,
        by default None
    **kwargs: Any
        Reconstruction parameters.  See :class:`~pycudasirecon.ReconParams` for valid
        keys.

    Returns
    -------
    np.ndarray
        The reconstructed array.
    """
    if otf_file is not None:
        otf = otf_file
        warnings.warn(
            'The "otf_file" argument is deprecated. Use "otf" instead.',
            DeprecationWarning,
        )

    if otf is not None:
        if psf is not None:
            warnings.warn("Both `otf` and `psf` provided, otf will be used.")
        if isinstance(otf, (str, Path)):
            if not Path(otf).exists():
                raise FileNotFoundError(f"Provided `otf` does not exist: {otf}")
            _otf_context: ContextManager = nullcontext(otf)
        else:
            _otf_context = temporary_otf(otf=otf)
    elif psf is not None:
        _otf_context = temporary_otf(psf=psf, **(makeotf_kwargs or {}))
    else:
        raise ValueError("Either `psf` or `otf` must be provided.")

    with _otf_context as _otf_path:
        with temp_config(otf_file=_otf_path, **kwargs) as cfg:
            return SIMReconstructor(array, cfg.name).get_result()


class SIMReconstructor:
    """Main class for SIM reconstruction.

    Parameters
    ----------
    arg0 : Union[np.ndarray, Tuple[int, int, int]]
        Either a numpy array of raw data, or a shape tuple (3 integers) that indicate
        the size of raw data (to be provided later with `set_raw`).
    config : str, optional
        Config file path.  Currently, the config file *must* include the otf-file key.
        by default None

    Raises
    ------
    ValueError
        If the array is not ndim==3 or shape is not a 3-tuple
    """

    def __init__(
        self,
        arg0: np.ndarray | tuple[int, int, int],
        config: str,
    ) -> None:
        image: np.ndarray | None
        if isinstance(arg0, np.ndarray):
            if arg0.ndim != 3:
                raise ValueError("array must have 3 dimensions")
            image = arg0
            self.shape = image.shape
        elif isinstance(arg0, (list, tuple)):
            if len(arg0) != 3:
                raise ValueError("shape argument must have length 3")
            image = None
            self.shape = arg0
        nz, ny, nx = self.shape
        with caplog():
            self._ptr = lib.SR_new_from_shape(nx, ny, nz, config.encode())

        if image is not None:
            self.set_raw(image)
            self.process_volume()

    def process_volume(self) -> None:
        """Process the current volume."""
        with caplog():
            lib.SR_processOneVolume(self._ptr)

    def set_raw(self, img: np.ndarray) -> None:
        """Assign `img` as the raw data to be reconstructed."""
        nz, ny, nx = img.shape
        if not np.issubdtype(img.dtype, np.float32) or not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img, dtype=np.float32)
        with caplog():
            lib.SR_setRaw(self._ptr, img, nx, ny, nz)
            lib.SR_loadAndRescaleImage(self._ptr, 0, 0)
            lib.SR_setCurTimeIdx(self._ptr, 0)

    def get_result(self) -> np.ndarray:
        """Return the current reconstruction result."""
        *_, ny, nx = self.shape
        rp = self.get_recon_params()
        nz = int(self.get_image_params().nz * rp.z_zoom)
        out_shape = (nz, int(ny * rp.zoomfact), int(nx * rp.zoomfact))
        _result = np.empty(out_shape, np.float32)
        lib.SR_getResult(self._ptr, _result)
        return _result

    # TODO: can't add type hints here yet since build-docs will try to resolve them
    # and that requires the lib still.
    def get_recon_params(self) -> lib.ReconParams:
        """Return the current reconstruction parameters."""
        return lib.ReconParams.from_address(lib.SR_getReconParams(self._ptr))

    def get_image_params(self) -> lib.ImageParams:
        """Return the current image parameters."""
        return lib.ImageParams.from_address(lib.SR_getImageParams(self._ptr))
