import contextlib
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile as tf
from pycudasirecon import ReconParams, make_otf, reconstruct, sim_reconstructor
from pycudasirecon._otf import temporary_otf

try:
    DATA = Path(__file__).parent / "data"
except NameError:
    DATA = Path("tests") / "data"

OTF = DATA / "otf.tif"
PSF = DATA / "psf.tif"
RAW = DATA / "raw.tif"
CONFIG = DATA / "config"
EXPECTED = tf.imread(DATA / "expected.tif")

ATOL = 0.3 if os.name == "nt" else 1e-08


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


def _close_enough(a, b, max_mean=0.3, max_std=0.06):
    diff = np.abs(a - b)
    return diff.mean() < max_mean and diff.std() < max_std


def test_reconstruct_with_file_otf(params):
    raw = tf.imread(RAW)
    result = reconstruct(raw, otf=OTF, **params.dict(exclude_unset=True))
    assert _close_enough(EXPECTED, result)


def test_reconstruct_with_array_otf(params):
    raw = tf.imread(RAW)
    result = reconstruct(raw, otf=tf.imread(OTF), **params.dict(exclude_unset=True))
    assert _close_enough(EXPECTED, result)


def test_reconstruct_with_file_psf(params):
    result = reconstruct(
        tf.imread(RAW),
        psf=PSF,
        makeotf_kwargs={"fixorigin": (3, 20)},
        **params.dict(exclude_unset=True),
    )
    assert _close_enough(EXPECTED, result)


def test_reconstruct_with_array_psf(params):
    result = reconstruct(
        tf.imread(RAW),
        psf=tf.imread(PSF),
        makeotf_kwargs={"fixorigin": (3, 20)},
        **params.dict(exclude_unset=True),
    )
    assert _close_enough(EXPECTED, result)


def test_make_otf_from_file(tmp_path):
    output = tmp_path / "_testotf.tif"
    make_otf(PSF, output, fixorigin=(3, 20))
    np.allclose(tf.imread(OTF), tf.imread(output))


def test_make_otf_from_array(tmp_path):
    output = tmp_path / "_testotf.tif"
    make_otf(tf.imread(PSF), output, fixorigin=(3, 20))
    np.allclose(tf.imread(OTF), tf.imread(output))


def test_temp_otf_from_file():
    with temporary_otf(PSF, fixorigin=(3, 20)) as temp_otf:
        assert np.allclose(tf.imread(OTF), tf.imread(temp_otf))
        assert Path(temp_otf).exists()
    assert not Path(temp_otf).exists()
