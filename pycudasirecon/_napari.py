from typing import TYPE_CHECKING, List, Tuple
from magicgui import magic_factory
from typing_extensions import Annotated
from napari.types import ImageData
from napari_plugin_engine import napari_hook_implementation
from pathlib import Path

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


@magic_factory(persist=True)
def SimReconstruct(
    image: "napari.layers.Image",
    otf: Path,
    nimm: float = 1.515,
    background: float = 0,
    wiener: Annotated[float, {"step": 0.001, "min": 0.001, "max": 0.01}] = 0.001,
    k0angles: Tuple[float, float, float] = (0.804300, 1.8555, -0.238800),
    ls: float = 0.2035,
    ndirs: int = 3,
    nphases: int = 5,
    na: float = 1.42,
    otfRA: bool = True,
    dampenOrder0: bool = True,
    xyres: float = 0.08,
    zres: float = 0.125,
    zresPSF: float = 0.125,
    pseudo_widefield: bool = True,
) -> "napari.types.LayerDataTuple":
    from pycudasirecon import reconstruct, ReconParams

    params = locals().copy()
    image = params.pop("image")
    _otf = str(params.pop("otf"))
    _params = ReconParams(**params)
    data = reconstruct(image.data, otf_file=_otf, **_params.dict(exclude_unset=True))
    meta = dict(
        scale=list(image.scale), name=image.name + " (recon)", blending="additive"
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
    return SimReconstruct
