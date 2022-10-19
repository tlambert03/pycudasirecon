from typing import TYPE_CHECKING, List, Tuple
from magicgui import magic_factory
from typing_extensions import Annotated
from napari.types import ImageData
from napari_plugin_engine import napari_hook_implementation
from pathlib import Path
import vispy.scene
import numpy as np

if TYPE_CHECKING:
    import numpy as np
    import napari.layers
    import napari.types


def _pseudo_wf(image: "np.ndarray", order: str = "azp", nphases=5, nangles=3):
    # TODO: deal with 4+ dims
    pax = order.index("p")
    aax = order.index("a")
    reshp = [-1, -1, -1] + list(image.shape[-2:])
    reshp[pax] = nphases
    reshp[aax] = nangles
    return image.reshape(reshp).mean((pax, aax))


def init(self):
    from qtpy.QtWidgets import QSizePolicy

    canvas = vispy.scene.SceneCanvas(keys="interactive")
    canvas.size = 800, 600
    view = canvas.central_widget.add_view(camera="panzoom")
    markers = vispy.scene.visuals.Markers(
        pos=np.array([(0, 0)]),
        size=10,
        scaling=True,
        parent=view.scene,
        alpha=0.2,
    )
    img = vispy.scene.visuals.Image(parent=view.scene, cmap="gray", clim=(0, 1))
    view.camera.aspect = 1  # type: ignore
    self.native.layout().addWidget(canvas.native)
    canvas.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @self.k0angles.changed.connect
    @self.ls.changed.connect
    def draw_spots():
        if img._data is None:
            return
        center = np.array(img._data.shape) / 2
        k0angles = self.k0angles.value
        ls = self.ls.value
        points = []
        scale = 1 / (ls * self.xyres.value)
        for angle in k0angles:
            pos = scale * np.array([np.cos(angle), np.sin(angle)])
            points.append(center + pos)
            points.append(center - pos)

        markers.set_data(np.array(points), face_color="blue", edge_color="blue")

    draw_spots()

    @self.image.changed.connect
    def on_image_change(new_image: "napari.layers.Image"):
        from scipy.fft import fftn, fftshift, ifftshift

        nz, ny, nx = new_image.data.shape
        angles = 3
        d = new_image.data.reshape(angles, -1, ny, nx)

        new = np.abs(ifftshift(fftn(fftshift(d), axes=(1, 2, 3)))).astype(np.float32)
        new = new.mean((0, 1))
        new -= new.min()
        new /= new.max() * 0.02
        img.set_data(new**0.4)
        view.camera.set_range(margin=0)


@magic_factory(persist=True, widget_init=init,)
def sim_reconstruct(
    image: "napari.layers.Image",
    otf: Path,
    nimm: float = 1.515,
    background: float = 0,
    wiener: Annotated[float, {"widget_type": "FloatSlider", "step": 0.001, "min": 0.001, "max": 0.02}] = 0.001,
    k0angles: Tuple[float, float, float] = (0.804300, 1.8555, -0.238800),
    ls: float = 0.2035,
    ndirs: int = 3,
    nphases: int = 5,
    na: float = 1.42,
    # otfRA: bool = True,
    dampenOrder0: bool = True,
    xyres: float = 0.08,
    zres: float = 0.125,
    zresPSF: float = 0.125,
    format: Annotated[str, {'choices': ['PZA', 'PAZ']}] = "PZA",

    pseudo_widefield: bool = True,
) -> "napari.types.LayerDataTuple":
    from pycudasirecon import reconstruct, ReconParams

    params = locals().copy()
    params['fastSI'] = bool(params.pop('format') != 'PZA')
    image = params.pop("image")
    _otf = str(params.pop("otf"))
    _params = ReconParams(**params)
    data = reconstruct(image.data, otf=_otf, **_params.dict(exclude_unset=True))
    meta = dict(scale=list(image.scale), name=f"{image.name} (recon)", blending="additive")

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
