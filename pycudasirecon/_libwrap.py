from ._ctyped import Library, CPointer
import numpy as np
from ctypes import c_short, c_ushort, c_int, c_float, c_long, c_uint, c_bool, Structure

try:
    lib = Library("libcudasirecon")
except FileNotFoundError:
    raise FileNotFoundError(
        "Unable to find library 'libcudasirecon'\n"
        "Please try `conda install -c conda-forge cudasirecon`."
    ) from None


@lib.function
def SR_new_from_shape(nx: int, ny: int, nImages: int, configFileName: str) -> CPointer:
    """Create new SIM_Reconstructor."""


@lib.function
def SR_setRaw(self: CPointer, raw_data: np.ndarray, nx: int, ny: int, nz: int) -> None:
    """Set raw data to reconstruct."""


@lib.function
def SR_loadAndRescaleImage(self: CPointer, it: int, iw: int) -> None:
    ...


@lib.function
def SR_setCurTimeIdx(self: CPointer, it: int) -> None:
    ...


@lib.function
def SR_processOneVolume(self: CPointer) -> None:
    ...


@lib.function
def SR_getResult(self: CPointer, result: np.ndarray) -> None:
    """Get result of reconstruction (populate `result` buffer)."""


@lib.function
def SR_getReconParams(self: CPointer) -> CPointer:
    """Get current reconstruction parameters"""


@lib.function
def SR_getImageParams(self: CPointer) -> CPointer:
    """Get current image parameters."""


# TODO
# @lib.function
# def SR_new_from_argv(argc: int, argv: List[str]) -> CPointer:
#     """Create new SIM_Reconstructor."""


@lib.function
def SR_setFile(self: CPointer, it: int, iw: int) -> None:
    ...


@lib.function
def SR_writeResult(self: CPointer, it: int, iw: int) -> None:
    ...


@lib.function
def SR_closeFiles(self: CPointer) -> None:
    ...


class ReconParams(Structure):
    _fields_ = [
        ("k0startangle", c_float),
        ("linespacing", c_float),
        ("na", c_float),
        ("nimm", c_float),
        ("ndirs", c_int),
        ("nphases", c_int),
        ("norders_output", c_int),
        ("norders", c_int),
        ("phaseSteps", c_long),
        ("bTwolens", c_int),
        ("bFastSIM", c_int),
        ("bBessel", c_int),
        ("BesselNA", c_float),
        ("BesselLambdaEx", c_float),
        ("deskewAngle", c_float),
        ("extraShift", c_int),
        ("bNoRecon", c_bool),
        ("cropXYto", c_uint),
        ("bWriteTitle", c_int),
        ("zoomfact", c_float),
        ("z_zoom", c_int),
        ("nzPadTo", c_int),
        ("explodefact", c_float),
        ("bFilteroverlaps", c_int),
        ("recalcarrays", c_int),
        ("napodize", c_int),
        ("bSearchforvector", c_int),
        ("bUseTime0k0", c_int),
        ("apodizeoutput", c_int),
        ("apoGamma", c_float),
        ("bSuppress_singularities", c_int),
        ("suppression_radius", c_int),
        ("bDampenOrder0", c_int),
        ("bFitallphases", c_int),
        ("do_rescale", c_int),
        ("equalizez", c_int),
        ("equalizet", c_int),
        ("bNoKz0", c_int),
        ("wiener", c_float),
        ("wienerInr", c_float),
        # there are more ...
    ]

    def __str__(self):
        return (
            f"<ReconParams ndirs={self.ndirs} nphases={self.nphases} "
            f"linespacing={self.linespacing:0.3f} ... >"
        )


class ImageParams(Structure):
    _fields_ = [
        ("nx", c_int),  # image's width after deskewing or same as "nx_raw"
        ("nx_raw", c_int),  # raw image's width before deskewing
        ("ny", c_int),
        ("nz", c_int),
        ("nz0", c_int),
        ("nwaves", c_short),
        ("wave0", c_short),
        ("wave1", c_short),
        ("wave2", c_short),
        ("wave3", c_short),
        ("wave4", c_short),
        ("ntimes", c_short),
        ("curTimeIdx", c_ushort),
        ("dxy", c_float),
        ("dz", c_float),
        ("dz_raw", c_float),
        ("inscale", c_float),
    ]

    def __str__(self):
        return (
            f"<ImageParams nz={self.nz} ny={self.ny} nx={self.nx} nt={self.ntimes} "
            f"dxy={self.dxy:0.3f} dz={self.dz:0.3f} ... >"
        )
