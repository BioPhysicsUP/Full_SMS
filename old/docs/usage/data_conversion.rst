Converting data to the Full SMS format
======================================

Full SMS uses a specific HDF format. For data in other formats, you will need to convert it to the Full SMS format.
For this purpose, there is a conversion tool built into the software that can convert data from ASCII as well as the
Photon-HDF5_ format [#]_. For other formats, the easiest solution is to first convert them to Photon-HDF5, see
:ref:`phconvert`.

To use this tool, open Full SMS, click on Tools -> Convert files. Next, select the file format you want to convert
from, the options are .pt3, .csv and .h5 (meaning Photon-HDF5).
The fundamental assumption of this tool is that each file contains one measurement (one particle).
.csv files need to be 2 columns -- "macro times" and "micro times", both in nanoseconds.
Choose the folder containing the files. The entire folder will be converted into a single Full SMS HDF5 file.
To convert multiple folders, select Bulk Convert, which will convert all subfolders into .h5 files.
Select the name pattern of the files, e.g. "trace" will match the files "trace 1.csv", "trace 2.csv" etc.
For .pt3 files, the channel can be specified.
Finally, choose the export file name and location. Click Convert to start the conversion.

.. _phconvert:
Converting data to Photon-HDF5
------------------------------

The Photon-HDF5 developers have created a Python library called phconvert to convert data from all major TCSPC formats
to Photon-HDF5. More information is on the Photon-HDF5 website linked above. As an example, here is a script to
convert measurements made in PicoQuant's SymPhoTime software to Photon-HDF5, which you may adapt to your needs::

    import os
    import numpy as np
    import phconvert as phc

    directory = '/your/measurement/directory'

    for root, dirnames, filenames in os.walk(directory):
        i = 0
        for filename in filenames:
            if filename.endswith('.ptu'):
                print(filename)
                i += 1
                fullfilename = os.path.join(root, filename)

                timestamps, detectors, nanotimes, meta = phc.pqreader.load_ptu(fullfilename)
                det0 = np.where(detectors == 0)
                timestamps = timestamps[det0]
                if nanotimes is not None:
                    nanotimes = nanotimes[det0]
                else:
                    print('No nanotimes found for particle!')
                    continue
                detectors = detectors[det0]

                description = 'Converted from PTU file: %s' % filename
                author = 'Albert Einstein'

                photon_data = dict(
                    timestamps=timestamps,
                    nanotimes=nanotimes,
                    detectors=detectors,
                    timestamps_specs={'timestamps_unit': meta['timestamps_unit']},
                    nanotimes_specs={'tcspc_unit': meta['nanotimes_unit']})

                setup = dict(
                    ## Mandatory fields
                    num_pixels=1,  # using 1 detector
                    num_spots=1,  # a single confocal excitation
                    num_spectral_ch=1,  # no donor and acceptor detection
                    num_polarization_ch=1,  # no polarization selection
                    num_split_ch=1,  # no beam splitter
                    modulated_excitation=True,  # pulsed excitation
                    excitation_alternated=[True],  # pulsed excitation
                    lifetime=True,  # TCSPC in detection
                )

                provenance = dict(
                    creation_time=meta['creation_time'],
                    filename=fullfilename
                )

                identity= dict(
                    author=author
                )

                data = dict(
                    description=description,
                    photon_data = photon_data,
                    setup=setup,
                    provenance=provenance,
                    identity=identity,
                    _filename=fullfilename
                )

                savename = f'particle{i}.h5'
                phc.hdf5.save_photon_hdf5(data, h5_fname=os.path.join(root, savename), overwrite=True)

.. _Photon-HDF5: https://photon-hdf5.org/
.. [#] Directly reading from Photon-HDF5 is a planned extension.