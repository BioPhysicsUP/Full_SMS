HDF5 File format for SMS
========================

HDF5_ is a file format designed for flexibility and the ability to storge large amounts of data.

.. _HDF5: https://en.wikipedia.org/wiki/Hierarchical_Data_Format

For our data, one hdf5 file is used for a dataset. It contains a number of groups corresponding to measured particles.
Each "particle" contains datasets for photon "microtimes", absolute arrival times, and spectra. It also contains the
raster scan as an array. Furthermore, the file itself, the particles and the datasets can have so-called attributes.

:doc:`listattr`



