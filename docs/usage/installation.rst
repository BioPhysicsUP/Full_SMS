Installation instructions
=========================

A Windows installer is available at https://github.com/BioPhysicsUP/Full_SMS/releases.
This should work without needing to install any additional requirements.

To install from source, Python 3 is required along with the following packages:

| PyQt5
| pyqtgraph
| numpy
| scipy
| matplotlib
| pandas
| h5py
| statsmodels
| pyarrow
| dill
| h5pickle

Python can be installed from https://www.python.org/downloads/.

To install a package, run this command in your terminal:

.. code-block:: console

    $ pip install <package>

where <package> is the name of the package (multiple packages can be passed at once, with spaces inbetween).

Full SMS can be downloaded using `git <https://git-scm.com/>`_:

.. code-block:: console

    $ git clone https://github.com/BioPhysicsUP/Full_SMS

Or by downloading the `zip <https://github.com/BioPhysicsUP/Full_SMS/archive/refs/heads/master.zip>`_.

After installing, navigate to the ``/src`` directory and run:

.. code-block:: console

    $ python main.py
