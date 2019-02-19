User's Guide
============

The main lifetime fitting code is in the module :mod:`tcspcfit`.

A fit is performed by creating an instance of a class. Currently the available classes are
:class:`tcspcfit.OneExp`, :class:`tcspcfit.TwoExp` and :class:`tcspcfit.ThreeExp`.

A very simple usage would be::

    fit1 = OneExp(irf, measured, t, channelwidth, ploton=True)

Where ``irf``, ``measured`` and ``t`` are all one-dimensional and the same size. ``channelwidth`` is the width of one
channel in nanoseconds. ``ploton=True`` instructs the fitting code to plot the result.

Now, ``fit1` is an object that contains all the information of the fit. This includes the results such as fitted
lifetimes, amplitudes, etc. as well as the raw data and all kinds of other things:


