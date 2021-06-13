# pycudasirecon

This package provides a python wrapper and convenience functions for
[cudasirecon](https://github.com/scopetools/cudasirecon), which is a CUDA/C++
implementation of Mats Gustafsson & Lin Shao's 3-beam SIM reconstruction software.
<sup>1</sup>.  It also supports lattice-light-sheet SIM (with deskewing).

Additionally, it includes a GPU implementation of the Hessian denoising algorithm
described in Huang 2008 <sup>2</sup>.

- 3D (3-beam) SIM reconstruction
- 2D (2-beam) SIM reconstruction, including TIRF-SIM
- Lattice-SIM reconstruction
- GPU-backed Hessian denoising

### Install

The conda package includes the required pre-compiled libraries for Windows and Linux. See GPU driver requirements [below](#gpu-requirements)

```sh
conda install -c conda-forge pycudasirecon
```

*macOS is not supported*

### 📖   &nbsp; [Documentation](http://www.talleylambert.com/pycudasirecon)


### GPU requirements

This software requires a CUDA-compatible NVIDIA GPU. The underlying cudasirecon
libraries have been compiled against different versions of the CUDA toolkit.
The required CUDA libraries are bundled in the conda distributions so you don't
need to install the CUDA toolkit separately.  If desired, you can pick which
version of CUDA you'd like based on your needs, but please note that different
versions of the CUDA toolkit have different GPU driver requirements:

To specify a specific cudatoolkit version, install as follows (for instance, to
use `cudatoolkit=10.2`)

```sh
conda install -c conda-forge pycudasirecon cudatoolkit=10.2
```

| CUDA | Linux driver | Win driver |
| ---- | ------------ | ---------- |
| 10.2 | ≥ 440.33     | ≥ 441.22   |
| 11.0 | ≥ 450.36.06  | ≥ 451.22   |
| 11.1 | ≥ 455.23     | ≥ 456.38   |
| 11.2 | ≥ 460.27.03  | ≥ 460.82   |


If you run into trouble, feel free to [open an
issue](https://github.com/tlambert03/pycudasirecon/issues) and describe your
setup.


## Quickstart

If you have a raw SIM image volume and a PSF and you just want to get started, check
out the `pycudasirecon.reconstruct` function, which should be able to handle most
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


___

#### References

<sup>1</sup> Gustafsson MG, Shao L, Carlton PM, Wang CJ, Golubovskaya IN, Cande WZ, Agard DA, Sedat JW. Three-dimensional resolution doubling in wide-field fluorescence microscopy by structured illumination. Biophys J. 2008 Jun;94(12):4957-70. doi: [10.1529/biophysj.107.120345](https://dx.doi.org/10.1529/biophysj.107.120345). Epub 2008 Mar 7. PMID: 18326650; PMCID: PMC2397368.

<sup>2</sup> Huang X, Fan J, Li L, Liu H, Wu R, Wu Y, Wei L, Mao H, Lal A, Xi P, Tang L, Zhang Y, Liu Y, Tan S, Chen L. Fast, long-term, super-resolution imaging with Hessian structured illumination microscopy. Nat Biotechnol. 2018 Jun;36(5):451-459. doi: [10.1038/nbt.4115](https://dx.doi.org/10.1038/nbt.4115). Epub 2018 Apr 11. PMID: 29644998.
