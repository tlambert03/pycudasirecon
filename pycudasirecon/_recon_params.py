from typing import Optional, Sequence

from pydantic import BaseModel, Field, FilePath


class ReconParams(BaseModel):
    otf_file: Optional[FilePath] = Field(None, description="OTF file")
    usecorr: bool = Field(
        False, description="use the flat-field correction file provided"
    )
    ndirs: int = Field(default=3, description="number of directions")
    nphases: int = Field(default=5, description="number of phases per direction")
    nordersout: int = Field(
        0, description="number of output orders; must be <= norders"
    )
    angle0: float = Field(1.648, description="angle of the first direction in radians")
    ls: float = Field(0.172, description="line spacing of SIM pattern in microns")
    na: float = Field(1.42, description="Detection numerical aperture")
    nimm: float = Field(1.515, description="refractive index of immersion medium")
    zoomfact: float = Field(2, description="lateral oversampling factor")
    explodefact: float = Field(
        1,
        description="artificially exploding the reciprocal-space "
        "distance between orders by this factor",
    )
    zzoom: int = Field(1, description="axial zoom factor")
    nofilteroverlaps: bool = Field(
        False,
        description="do not filter the overlaping region between bands "
        "usually used in trouble shooting",
    )
    background: float = Field(0, description="camera readout background")
    wiener: float = Field(0.01, description="Wiener constant")
    forcemodamp: Optional[Sequence[float]] = Field(
        None, description="modamps forced to these values"
    )
    k0angles: Optional[Sequence[float]] = Field(
        None, description="user given pattern vector k0 angles for all directions"
    )
    otfRA: bool = Field(True, description="using rotationally averaged OTF")
    otfPerAngle: bool = Field(True, description="using one OTF per SIM angle")
    fastSI: bool = Field(
        True,
        description="SIM data is organized in Z->Angle->Phase order; "
        "default being Angle->Z->Phase",
    )
    k0searchAll: bool = Field(False, description="search for k0 at all time points")
    norescale: bool = Field(False, description="bleach correcting for z")  # TODO
    equalizez: bool = Field(True, description="bleach correcting for z")
    equalizet: bool = Field(True, description="bleach correcting for time")
    dampenOrder0: bool = Field(True, description="dampen order-0 in final assembly")
    nosuppress: bool = Field(
        False,
        description="do not suppress DC singularity in final assembly "
        "(good idea for 2D/TIRF data)",
    )
    nokz0: bool = Field(
        True, description="do not use kz=0 plane of the 0th order in the final assembly"
    )
    gammaApo: float = Field(
        1, description="output apodization gamma; 1.0 means triangular apo"
    )
    bessel: bool = Field(False, description="bessel-SIM data")
    besselExWave: float = Field(
        0.488, description="Bessel SIM excitation wavelength in microns"
    )
    besselNA: float = Field(0.144, description="Bessel SIM excitation NA)")
    deskew: float = Field(
        0,
        description="Deskew angle; if not 0.0 then perform deskewing before processing",
    )
    deskewshift: int = Field(
        0,
        description="If deskewed, the output image's extra shift in X (positive->left)",
    )
    noRecon: bool = Field(
        False,
        description="No reconstruction will be performed; "
        "useful when combined with --deskew",
    )
    cropXY: int = Field(
        0, description="Crop the XY dimension to this number; 0 means no cropping"
    )
    xyres: float = Field(0.1, description="XY pixel size")
    zres: float = Field(0.2, description="Z step size")
    zresPSF: float = Field(0.15, description="Z step size of the PSF")
    wavelength: int = Field(530, description="emission wavelength in nanometers")
    writeTitle: bool = Field(
        False,
        description="Write command line to image header "
        "(may cause issues with bioformats)",
    )

    def to_config(self, exclude_unset=True):
        lines = []
        for k, v in self.dict(exclude_unset=exclude_unset).items():
            if k == "k0angles":
                v = ",".join(str(x) for x in v)
            if isinstance(v, bool):
                v = int(v)
            lines.append(f'{k.replace("_", "-")}={v}')
        return "\n".join(lines)
