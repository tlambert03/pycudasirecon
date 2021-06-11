# pycudasirecon

This package provides a python wrapper and convenience functions for
[cudasirecon](https://github.com/scopetools/cudasirecon), which is a CUDA/C++
implementation of Mats Gustafsson & Lin Shao's 3-beam SIM reconstruction software.
{cite}`Gustafsson2008`.  It also supports lattice-light-sheet SIM (with deskewing).

Additionally, it includes a GPU implementation of the Hessian denoising algorithm
described in Huang 2008 {cite}`Huang2018`.

- 3D (3-beam) SIM reconstruction
- 2D (2-beam) SIM reconstruction, including TIRF-SIM
- Lattice-SIM reconstruction
- GPU-backed Hessian denoising

## Install

Install (Linux and Windows) from conda forge:

```bash
conda install -c conda-forge pycudasirecon
```

see GPU requirements in [Installation](installation.md).

## Quickstart

If you have a PSF and an image volume and you just want to get started, check
out the {func}`pycudasirecon.reconstruct` function, which should be able to handle most
basic applications.

```python
from pycudasirecon import reconstruct

raw = tf.imread('path/to/raw_data.tif')
psf = tf.imread('path/to/sim_psf.tif')
makeotf_kwargs = {}  # kwargs for pycudasirecon.make_otf
recon_params = {}  # kwargs for pycudasirecon.ReconParams
result = reconstruct(
    raw,
    psf=psf,
    makeotf_kwargs=makeotf_kwargs,
    **recon_params
)
```

This library is in development ... more details to follow.

## References

```{bibliography}
:style: unsrt
```
