# surfh

SUper Resolution and Fusion for Hyperspectral images

Package for processing of MIRI MRS data.

## Installation 

`poetry install`

#### Set-up GPU



## Usage

To use the MRS Linear-mixing Model fusion scipts, First create the following directories :

- Fusion
  - Corrected_slices
  - Filtered_slices
  - Masks
  - PSF
  - Raw_slices
  - Results
  - Templates


### Real data pre-processing

To correct the distortion of data from stage 2 of the JWST pipeline, run :

`python script/correction_mrs_data.py`

Before running the script, remember to change the path to `Fusion/Raw_slices` et `Fusion/Corrected_slices`.

This script correct all slices from `Fusion/Raw_slices` and save them to `Fusion/Corrected_slices`.


Afterward, filter data using :

`python script/filter_corrected_mrs_data.py`

Before running the script, remember to change the path to `Fusion/Corrected_slices` et `Fusion/Filtered_slices`.

This script applies a median filter to all corrected slices in order to remove all spectral lines.


### Real data fusion




## Data vizualisation

## TODO

- Parallelization on channel
- Use HPC tools like cython, pythran, numba, mpi4py for intra-node