Full SMS HDF5 file specification (Version 1.08)
===============================================

HDF5 file
---------
Attributes
''''''''''
'# Particles' (int):
    The number of particles in the file.
'Version' (string):
    File version.

Objects
'''''''
'Particle <#>' (HDF5 Group):
    An object corresponding to the measurement of a particle, <#> is a 1-indexed integer.
    See `Particle group`_ below.

Particle group
--------------
Attributes
''''''''''
'Date' (string):
    The date of the measurement. uses the format "Tuesday, June 27, 2023 11:22 AM"
'Description'(string):
    A general description of the measurement.
'Has Power Measurement?' (bool):
    Whether the laser power was measured and saved for this particle.
'Intensity?' (int):
    Whether intensity (photon arrival times) was measured for this particle.
'RS Coord. (um)' (size 2 array of 64-bit float):
    The raster scan coordinates fo the particle (x, y).
'Spectra?' (int):
    Whether a spectral time trace was measured for this particle.
'User' (string):
    The name of the user who performed the measurement.

Objects
'''''''
'Absolute Times (ns)' (HDF5 dataset (array of int)):
    Absolute photon arrival times. Has `Photon times`_ attributes.
'Absolute Times 2 (ns)' (HDF5 dataset (array of int)):
    Absolute photon arrival times in second channel. Has `Photon times`_ attributes.
'Intensity trace (cps)' (HDF5 dataset (2D array of float)):
    Time-binned intensity values. Time is in seconds, intensity is in counts per second. Has no attributes.
'Micro Times (ns)' (HDF5 dataset (array of float)):
    Relative photon arrival times. Has `Photon times`_ attributes.
'Micro Times 2 (ns)' (HDF5 dataset (array of float)):
    Relative photon arrival times in second channel. Has `Photon times`_ attributes.
'Raster Scan' (HDF5 dataset (array of float)):
    Raster scan image array. Has `Raster Scan`_ attributes.
'Spectra (counts\s)' (HDF5 dataset (array of float)):
    Spectral time trace (wavelength x time). Has `Spectra`_ attributes.

Dataset attributes
------------------
Photon times
''''''''''''
'# Photons' (int):
    Number of photon times in the dataset, i.e. size of the dataset.
'bh Card' (string):
    Name of the TCSPC channel, e.g. the name of the card used for that channel or could be any other feature
    distinguishing the two channels.

Raster Scan
'''''''''''
'Int. Time (ms/um)' (float):
    The integration time for the raster scan in milliseconds/micrometer.
'Pixels per Line' (int):
    The number of pixels per line of the raster scan (also the number of lines since the scan is square).
'Range (um)' (float):
    The linear size of the raster scan in both dimensions, in micrometer.
'XStart (um)' (float)
    The microscope stage start X position (lower left of scan).
'YStart (um)' (float)
    The microscope stage start Y position (lower left of scan).
'bh Card' (string):
    Name of the TCSPC channel used for the scan, similar to the attribute in `Photon times`_.

Spectra
'''''''
'Exposure Time (s)' (float):
    CCD exposure time.
'Spectra Abs. Times (s)' (array of float):
    Time steps of recorded spectra.
'Wavelengths' (array of float):
    Wavelengths of measured spectral data in nm.


