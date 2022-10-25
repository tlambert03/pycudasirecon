from email.utils import quote
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Annotated


import numpy as np
import vispy.scene
from magicgui import magic_factory
from napari.types import ImageData
from napari_plugin_engine import napari_hook_implementation
from scipy.fft import fftn, fftshift, ifftshift, rfftn
from typing_extensions import Annotated
from napari.components import ViewerModel
from napari.qt import QtViewer


if TYPE_CHECKING:
    import napari.layers
    import napari.types
    import numpy as np


def _split_paz(
    image: "np.ndarray", order: str = "pza", nphases=5, nangles=3
) -> np.ndarray:
    """Return 5 dimensional array with shape (nphases, nangles, nz, ny, nx)"""
    assert image.ndim == 3, "image must be 3D"
    _order = list(reversed(order.lower()))
    pax = _order.index("p")
    aax = _order.index("a")
    zax = _order.index("z")
    reshp = [-1, -1, -1] + list(image.shape[-2:])
    reshp[pax] = nphases
    reshp[aax] = nangles
    img = image.reshape(reshp)
    return img.transpose((pax, aax, zax, -2, -1))


def _fft(data: np.ndarray, r: bool = False, **kwargs):
    return ifftshift((rfftn if r else fftn)(fftshift(data), **kwargs))


def _pseudo_wf(image: "np.ndarray", order: str = "pza", nphases=5, nangles=3):
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
    return fpaz.sum(axis=1)


def build_rgb_fft(data: np.ndarray, order="pza", nphases=5, nangles=3):
    """Return RGB image of FFT of data"""
    pazyx = _split_paz(data, order, nphases, nangles)
    orders = _extract_orders(pazyx)
    rgb = orders.sum(axis=0).transpose()
    rgb -= rgb.min()
    rgb /= rgb.max()
    return rgb**0.5


def init(self):
    from qtpy.QtWidgets import QSizePolicy

    for wdg in self.k0angles:
        wdg.min = -np.pi
        wdg.max = np.pi
    
    self._fft_viewer = ViewerModel()
    self._fft_qtviewer = QtViewer(self._fft_viewer)
    img: napari.layers.Image = self._fft_viewer.add_image(np.zeros((128, 128, 3)))
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
    def draw_spots():
        if img.data is None:
            return

        center = np.array(img.data.shape[:-1]) / 2
        _points = []
        scale = 0.8 / (self.ls.value * self.xyres.value)
        for angle in self.k0angles.value:
            pos = scale * np.array([np.cos(angle), np.sin(angle)])
            _points.append(center + pos)
            _points.append(center - pos)

        points.data = np.array(_points)
        points.edge_color = ["red", "red", "green", "green", "blue", "blue"]
        points.selected_data = {}

    @self.image.changed.connect
    def on_image_change(new_image: "napari.layers.Image"):

        rgb = build_rgb_fft(
            new_image.data,
            self.format.value,
            self.nphases.value,
            len(self.k0angles.value),
        )
        img.data = rgb
        draw_spots()
        self._fft_viewer.reset_view()

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
    meta = dict(
        scale=list(image.scale), name=f"{image.name} (recon)", blending="additive"
    )

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


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return sim_reconstruct
