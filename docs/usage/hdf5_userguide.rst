HDF5 File format for SMS
========================

.. toctree::
    :hidden:

    listattr

HDF5_ is a file format designed for flexibility and the ability to storge large amounts of data.

.. _HDF5: https://en.wikipedia.org/wiki/Hierarchical_Data_Format

In the Full SMS HDF5 format, one HDF5 file is used for a dataset. It contains a number of groups corresponding
to measured particles, like this::

    filename.h5
        ├ Particle 1
        ├ Particle 2
        ├ Particle 3
        ├ ..

Each "particle" contains datasets for photon "microtimes", absolute arrival times, and spectra. It also contains the
raster scan as an array::

    filename.h5
        ├ Particle 1
        |   ├ Absolute Times (ns)
        |   ├ Absolute Times 2 (ns)
        |   ├ Intensity Trace (cps)
        |   ├ Micro Times (ns)
        |   ├ Micro Times 2 (ns)
        |   ├ Raster Scan
        |   ├ Spectra (counts\s)
        ├ Particle 2
        |   ├ Absolute Times (ns)
        |   ├ Absolute Times 2 (ns)
        |   ├ Intensity Trace (cps)
        |   ├ Micro Times (ns)
        |   ├ Micro Times 2 (ns)
        |   ├ Raster Scan
        |   ├ Spectra (counts\s)
        ├ Particle 3
        |   ├ ..

Furthermore, the file itself, the particles and the datasets can have so-called attributes. A full list of attributes
can be found in  :doc:`listattr`. As changes need to be made to the file format specification from time to time, a
versioning system is used to ensure that older measurement files can be read correctly by *Full SMS*. The current
version is 1.08. Documentation for older versions will be added to this wiki in due course.

The module used for reading files is :mod:`smsh5_file_reader`.



