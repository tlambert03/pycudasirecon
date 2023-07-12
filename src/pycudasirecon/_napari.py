import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
from magicgui import magic_factory
from napari.components import ViewerModel
from napari.qt import QtViewer
from scipy.fft import fftfreq, fftn, fftshift, ifftshift, rfftn
from typing_extensions import Annotated

if TYPE_CHECKING:
    import napari.layers
    import napari.types


@functools.lru_cache(maxsize=None)
def _ffreqs(shape: int, spacing: int) -> np.ndarray:
    return 1 / (fftfreq(shape, spacing) + np.finfo(float).eps)


def _split_paz(
    image: "np.ndarray", order: str = "pza", nphases: int = 5, nangles: int = 3
) -> np.ndarray:
    """Return 5 dimensional array with shape (nphases, nangles, nz, ny, nx)."""
    if image.ndim != 3:
        raise ValueError("image must be 3D")
    _order = list(reversed(order.lower()))
    pax = _order.index("p")
    aax = _order.index("a")
    zax = _order.index("z")
    reshp = [-1, -1, -1, *list(image.shape[-2:])]
    reshp[pax] = nphases
    reshp[aax] = nangles
    img = image.reshape(reshp)
    return img.transpose((pax, aax, zax, -2, -1))


def _fft(data: np.ndarray, r: bool = False, **kwargs: Any) -> np.ndarray:
    return ifftshift((rfftn if r else fftn)(fftshift(data), **kwargs))


def _pseudo_wf(
    image: "np.ndarray", order: str = "pza", nphases: int = 5, nangles: int = 3
) -> np.ndarray:
    # TODO: deal with 4+ dims
    return _split_paz(image, order, nphases, nangles).mean((0, 1))


def _extract_orders(pazyx: np.ndarray) -> np.ndarray:
    """Return array with shape ((nphases-1)/2, nangles, ny, nx).

    the first axis highlights the fft orders. from high frequency to low.
    the second axis is the various angles
    """
    # shows components well:
    # take fft over PYX
    fpaz = np.abs(_fft(pazyx, axes=(-5, -2, -1)))
    fpaz = np.delete(fpaz, 3, 0)  # removes the unshifted component
    fpaz = fpaz.mean(axis=2)  # average over z
    fpaz = fpaz.reshape((2, 2, 3, *fpaz.shape[-2:]))
    return fpaz[:, 1]  # adds the mid freq and high freq together


def build_rgb_fft(
    data: np.ndarray, order: str = "pza", nphases: int = 5, nangles: int = 3
) -> np.ndarray:
    """Return RGB image of FFT of data."""
    pazyx = _split_paz(data, order, nphases, nangles)
    orders = _extract_orders(pazyx)
    rgb = orders.sum(axis=0).transpose()
    rgb -= rgb.min()
    rgb /= rgb.max()
    return rgb**0.6


def init(self: Any) -> None:
    from qtpy.QtWidgets import QSizePolicy

    for wdg in self.k0angles:
        wdg.min = -np.pi
        wdg.max = np.pi
        wdg.native.setDecimals(3)
    self.ls.native.setDecimals(3)

    self._fft_viewer = ViewerModel()
    self._fft_qtviewer = QtViewer(self._fft_viewer)
    img: napari.layers.Image = self._fft_viewer.add_image(
        np.zeros((128, 128, 3)), rgb=True, contrast_limits=(0, 0.7)
    )
    points: napari.layers.Points = self._fft_viewer.add_points(
        face_color="transparent",
        opacity=0.7,
    )

    nativecan = self._fft_qtviewer.canvas.native
    nativecan.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.native.layout().addWidget(nativecan)
    nativecan.show()
    self.max_width = 600

    @self.k0angles.changed.connect
    @self.ls.changed.connect
    @self.xyres.changed.connect
    def draw_spots() -> None:
        if img.data is None:
            return

        ny, nx, _ = img.data.shape
        center = np.array([ny, nx]) / 2
        freqs = _ffreqs(nx, self.xyres.value)
        # ls * 2 because the inner spot is much easier to see.
        scale = np.argmax((self.ls.value * 2) > freqs)

        _points = [[np.cos(a), np.sin(a)] for a in self.k0angles.value]
        points.data = center + scale * np.array(_points)
        # points.edge_color = ["red", "red", "green", "green", "blue", "blue"]
        points.edge_color = ["red", "green", "blue"]
        points.selected_data = {}

    @self.image.changed.connect
    @self.format.changed.connect
    @self.nphases.changed.connect
    def on_image_change() -> None:
        if self.image.value is None:
            return

        rgb = build_rgb_fft(
            self.image.value.data,
            self.format.value,
            self.nphases.value,
            len(self.k0angles.value),
        )
        img.data = rgb
        draw_spots()
        self._fft_viewer.reset_view()

    on_image_change()


@magic_factory(persist=True, widget_init=init)
def sim_reconstruct(
    image: "napari.layers.Image",
    otf: Path,
    nimm: float = 1.515,
    background: float = 0,
    wiener: Annotated[
        float, {"widget_type": "FloatSlider", "step": 0.001, "min": 0.001, "max": 0.02}
    ] = 0.001,
    k0angles: Tuple[float, float, float] = (0.804300, 1.8555, -0.238800),
    ls: float = 0.2035,
    ndirs: int = 3,
    nphases: int = 5,
    na: float = 1.42,
    xyres: float = 0.08,
    zres: float = 0.125,
    zresPSF: float = 0.125,
    format: Annotated[str, {"choices": ["PZA", "PAZ"]}] = "PZA",
    # otfRA: bool = True,
    dampenOrder0: bool = True,
    pseudo_widefield: bool = True,
) -> "napari.types.LayerDataTuple":
    from pycudasirecon import ReconParams, reconstruct

    params = locals().copy()
    params["fastSI"] = bool(params.pop("format") != "PZA")
    image = params.pop("image")
    _otf = str(params.pop("otf"))
    _params = ReconParams(otfRA=True, **params)
    data = reconstruct(image.data, otf=_otf, **_params.dict(exclude_unset=True))
    meta: dict = {
        "scale": list(image.scale),
        "name": f"{image.name} (recon)",
        "blending": "additive",
    }

    meta["scale"][-1] = meta["scale"][-1] / 2
    meta["scale"][-2] = meta["scale"][-2] / 2
    meta["translate"] = [0, -0.25, -0.25]  # FIXME
    out = [(data, meta)]
    if pseudo_widefield:
        meta["colormap"] = "green"
        out.append(
            (
                _pseudo_wf(image.data),
                {"scale": image.scale, "colormap": "magenta", "blending": "additive"},
            )
        )
    return out
