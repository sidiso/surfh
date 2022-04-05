# surfh

Package for processing of MIRI MRS data.

NOT USABLE at that time.

## Data format

To use `models.SpectroLMM`, the data must be in this format.

- `models.SpectroLMM` use a list of `ifu.Instr` and generate internally a
  corresponding list of `models.Channel`.
- Each `models.Channel` consider that data are in a numpy.array with shape (pointing, slit, wavelength, alpha). The precise shape is `models.Channel.oshape`.
- Therefor, for `models.SpectroLMM` the data is an array of shape (N, ) where each data of `models.Channel` are ravel() and concatenated in the order of the list.

## TODO

- Custom interpolation code for better performance (cython ?)
- Try precomputation of otf with LMM model (like in tci2022)
- Parallelization on channel
- Parallelization on pointing

- Use HPC tools like cython, pythran, numba for intra-node
- https://flothesof.github.io/optimizing-python-code-numpy-cython-pythran-numba.html
- https://jochenschroeder.com/blog/articles/DSP_with_Python2/
- mpi is also a possiblity for inter-node
