import contextlib
from pathlib import Path

import numpy as np
import pytest
import tifffile as tf

from pycudasirecon import ReconParams, reconstruct, sim_reconstructor

DATA = Path(__file__).parent / "data"
OTF = DATA / "otf.tif"
RAW = DATA / "raw.tif"
CONFIG = DATA / "config"
EXPECTED = DATA / "expected.tif"


@pytest.fixture(autouse=True)
def nocap(monkeypatch):
    """Fixes pytest while still letting us capture stdout with caplog."""
    noop = getattr(contextlib, "nullcontext", contextlib.suppress)
    monkeypatch.setattr(sim_reconstructor, "caplog", noop)


@pytest.fixture
def params():
    return ReconParams(
        nimm=1.515,
        background=0,
        wiener=0.001,
        k0angles=(0.804300, 1.8555, -0.238800),
        ls=0.2035,
        ndirs=3,
        nphases=5,
        na=1.42,
        otfRA=1,
        dampenOrder0=1,
        xyres=0.08,
        zres=0.125,
        zresPSF=0.125,
    )


def test_reconstruct(params):
    raw = tf.imread(RAW)
    expected = tf.imread(EXPECTED)
    result = reconstruct(raw, otf_file=str(OTF), **params.dict(exclude_unset=True))
    tf.imsave("out.tif", result, imagej=True)
    assert np.allclose(expected, result)
