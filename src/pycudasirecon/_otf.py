from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Union

import numpy as np
import tifffile

MAKEOTF = "makeotf"  # binary name for subprocess
PathLike = Union[str, Path]


def make_otf(  # noqa: C901
    psf: np.ndarray | PathLike,
    out_file: str,
    nphases: int = 5,
    beaddiam: float = 0.12,
    angle: float = 1.57,
    ls: float = 0.2,
    nocompen: bool = False,
    fixorigin: tuple[int, int] = (2, 9),
    na: float = 1.4,
    nimm: float = 1.515,
    leavekz: tuple[int, int, int] = (0, 0, 0),
    background: int | None = None,
) -> None:
    """Make OTF file from PSF.

    Note: currently there is no direct C-buffer access when making an OTF... so data
    must be read from and written to actual files. Providing psf as `path/to/psf.tif`
    will be faster.

    Parameters
    ----------
    psf : array-like, or str or Path
        The PSF to convert to an OTF.  Currently faster to just provide a path to a
        file, but array-input is accepted as a convenience.
    out_file : str
        File path to save to
    nphases : int, optional
        Number of phases, by default 5
    beaddiam : float, optional
        The diameter of the bead in microns, by default 0.12
    angle : float, optional
        The k0 vector angle with which the PSF is taken, by default 0
    ls : float, optional
        The illumination pattern's line spacing in microns, by default 0.2
    nocompen : bool, optional
        Do not perform bead size compensation, default False (do perform)
    fixorigin : Tuple[int, int], optional
        The starting and end pixel for interpolation along kr axis, by default (2, 9)
    na : float, optional
        The (effective) NA of the objective, by default 1.4
    nimm : float, optional
        The index of refraction of the immersion liquid, by default 1.515
    leavekz : Tuple[int, int, int], optional
        Pixels to be retained on kz axis, by default (0, 0, 0)
    background : Optional[int], optional
        User-supplied number as the background to subtract. If `None`, background
        will be estimated from image, by default `None`
    """
    kwargs = locals().copy()

    temp_psf = None
    if not isinstance(psf, (str, Path)):
        temp_psf = NamedTemporaryFile(suffix=".tif", delete=False)
        temp_psf.close()
        tifffile.imwrite(temp_psf.name, psf)
        psf_path = temp_psf.name
    elif Path(psf).exists():
        psf_path = str(psf)
    else:
        raise FileNotFoundError(f"Could not find psf at path: {psf}")

    # convert local kwargs to `makeotf` compatible arguments
    cmd = [MAKEOTF, psf_path, str(out_file)]
    for key, value in kwargs.items():
        if key in ("psf", "out_file"):
            continue
        if key == "background" and value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"-{key}")
            continue
        cmd.append(f"-{key}")
        if isinstance(value, (tuple, list)):
            cmd.extend([str(v) for v in value])
        else:
            cmd.append(str(value))

    try:
        run(cmd)
    finally:
        if temp_psf is not None:
            Path(temp_psf.name).unlink(missing_ok=True)


@contextmanager
def temporary_otf(
    psf: np.ndarray | PathLike | None = None,
    otf: np.ndarray | None = None,
    **kwargs: Any,
) -> Iterator[str]:
    """Convenience to generate a temporary OTF file from a PSF file or array."""
    if (psf is None) and (otf is None):
        raise ValueError("Must provide either psf or otf")

    temp_otf = NamedTemporaryFile(suffix=".tif", delete=False)
    temp_otf.close()
    kwargs["out_file"] = temp_otf.name
    try:
        if psf is not None:
            make_otf(psf, **kwargs)
        else:
            tifffile.imwrite(temp_otf.name, otf, photometric="MINISBLACK")
        yield temp_otf.name
    finally:
        Path(temp_otf.name).unlink()
