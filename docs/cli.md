# Command Line Interface

If you installed `pycudasirecon` using conda, the original binaries for OTF
generation (`makeotf`) and sim reconstruction (`cudasirecon`) will also be
installed in your conda environment.

## cudasirecon

The `cudasirecon` command runs sim reconstruction:

### Examples

```bash
 cudasirecon . 488nm _otf.tif --xyres=0.08 --zres=0.125 --zresPSF=0.125 -c config
```

Run `cudasirecon --help` at the command prompt for the full menu of options

```text
$ cudasirecon -h
cudasirecon v1.1.0

  --input-file arg                  input file (or data folder in TIFF mode)
  --output-file arg                 output file (or filename pattern in TIFF
                                    mode)
  --otf-file arg                    OTF file
  --usecorr arg                     use the flat-field correction file provided
  --ndirs arg (=3)                  number of directions
  --nphases arg (=5)                number of phases per direction
  --nordersout arg (=0)             number of output orders; must be <= norders
  --angle0 arg (=1.648)             angle of the first direction in radians
  --ls arg (=0.172000006)           line spacing of SIM pattern in microns
  --na arg (=1.20000005)            Detection numerical aperture
  --nimm arg (=1.33000004)          refractive index of immersion medium
  --zoomfact arg (=2)               lateral zoom factor
  --explodefact arg (=1)            artificially exploding the reciprocal-space
                                    distance between orders by this factor
  --zzoom arg (=1)                  axial zoom factor
  --nofilteroverlaps [=arg(=0)]     do not filter the overlaping region between
                                    bands usually used in trouble shooting
  --background arg (=0)             camera readout background
  --wiener arg (=0.00999999978)     Wiener constant
  --forcemodamp arg                 modamps forced to these values
  --k0angles arg                    user given pattern vector k0 angles for all
                                    directions
  --otfRA [=arg(=1)]                using rotationally averaged OTF
  --otfPerAngle [=arg(=1)]          using one OTF per SIM angle
  --fastSI [=arg(=1)]               SIM data is organized in Z->Angle->Phase
                                    order; default being Angle->Z->Phase
  --k0searchAll [=arg(=0)]          search for k0 at all time points
  --norescale [=arg(=0)]            bleach correcting for z
  --equalizez [=arg(=1)]            bleach correcting for z
  --equalizet [=arg(=1)]            bleach correcting for time
  --dampenOrder0 [=arg(=1)]         dampen order-0 in final assembly
  --nosuppress [=arg(=0)]           do not suppress DC singularity in final
                                    assembly (good idea for 2D/TIRF data)
  --nokz0 [=arg(=1)]                do not use kz=0 plane of the 0th order in
                                    the final assembly
  --gammaApo arg (=1)               output apodization gamma; 1.0 means
                                    triangular apo
  --saveprefiltered arg             save separated bands (half Fourier space)
                                    into a file and exit
  --savealignedraw arg              save drift-fixed raw data (half Fourier
                                    space) into a file and exit
  --saveoverlaps arg                save overlap0 and overlap1 (real-space
                                    complex data) into a file and exit
  -c [ --config ] arg               name of a file of a configuration.
  --2lenses [=arg(=1)]              I5S data
  --bessel [=arg(=1)]               bessel-SIM data
  --besselExWave arg (=0.488000005) Bessel SIM excitation wavelength in microns
  --besselNA arg (=0.143999994)     Bessel SIM excitation NA)
  --deskew arg (=0)                 Deskew angle; if not 0.0 then perform
                                    deskewing before processing
  --deskewshift arg (=0)            If deskewed, the output image's extra shift
                                    in X (positive->left)
  --noRecon                         No reconstruction will be performed; useful
                                    when combined with --deskew
  --cropXY arg (=0)                 Crop the X-Y dimension to this number; 0
                                    means no cropping
  --xyres arg (=0.100000001)        x-y pixel size (only used for TIFF files)
  --zres arg (=0.200000003)         z pixel size (only used for TIFF files)
  --zresPSF arg (=0.150000006)      z pixel size used in PSF TIFF files)
  --wavelength arg (=530)           emission wavelength (only used for TIFF
                                    files)
  --writeTitle [=arg(=1)]           Write command line to image header (may
                                    cause issues with bioformats)
  -h [ --help ]                     produce help message
  --version                         show version
```

## makeotf

The `makeotf` command turns a 3D PSF volume into a radially-averaged 2D
complex OTF file that can be used by cudasirecon.

### Examples

```bash
makeotf psf.tif _otf.tif -nphases 5 -fixorigin 3 20
```

Run `makeotf -h` at the command prompt for the full menu of options

```text
$ makeotf -h

  Options:
    -nphases -- number of phases; default 5
    -beaddiam F -- the diameter of the bead in microns (default is 0.12)
    -angle F -- the k0 vector angle with which the PSF is taken (default is 0)
    -ls x -- the illumination pattern's line spacing (in microns, default is 0.2)
    -nocompen -- do not perform bead size compensation (default is to perform)
    -5bands -- to output 5 OTF bands (default is combining higher-order's real and imag bands into one output
    -fixorigin kr1 kr2 -- the starting and end pixel for interpolation along kr axis (default is 2 and 9)
    -na x -- the (effective) NA of the objective
    -nimm x -- the index of refraction of the immersion liquid
    -leavekz kz1_1 kz1_2 kz2 -- the pixels to be retained on kz axis
    -I2M I2m_otf_file_name -- data contains I2M PSF and supply the I2M otf file name
    -background bkgd_value -- use the supplied number as the background to subtract
    -help or -h -- print this message
```
