User's Guide: Lifetime fitting
==============================

The main lifetime fitting code is in the module :mod:`tcspcfit`.

A fit is performed by creating an instance of a class. Currently the available classes are
:class:`tcspcfit.OneExp`, :class:`tcspcfit.TwoExp` and :class:`tcspcfit.ThreeExp`.

A very simple usage would be::

    from tcspcfit import *
    fit1 = OneExp(irf, measured, t, channelwidth, ploton=True)

Where ``irf``, ``measured`` and ``t`` are all one-dimensional and the same size. ``channelwidth`` is the width of one
channel in nanoseconds. ``ploton=True`` instructs the fitting code to plot the result.

Now, ``fit1`` is an object that contains all the information of the fit. This includes the results such as fitted
lifetimes, amplitudes, etc. as well as the raw data and all kinds of other things. A few important properties:

- ``fit1.tau`` is the lifetimes
- ``fit1.amp`` is the amplitudes
- ``fit1.shift`` is the IRF shift
- ``fit1.chisq`` is the chi-squared value
- ``fit1.bg`` is the decay background value
- ``fit1.irfbg`` is the IRF background value

Fits generally require initial values::

    fit1 = OneExp(irf, measured, t, channelwidth, tau=[1.5, 0.1, 10, 0],
                  shift=[0, -100, 100, 0], ploton=True)

Where for example ``tau=[1.5, 0.1, 10, 0]`` means: starting value 1.5, minimum value 0.1, maximum value 10, and the
final 0 means that the value should not be fixed (it would be a 1 if it should be fixed. If the parameter value is
fixed, the min and max values are ignored).

The IRF shift units are number of channels.

A two exponential fit has amplitudes as parameters as well (note how multiple parameters are specified)::

    tau = [[3, 0.01, 10, 0],
          [0.1, 0.01, 10, 0]]

    shift = [0, -100, 100, 0]

    amp = [[1, 0.01, 100, 0],
          [1, 0.01, 100, 0]]

    fit = TwoExp(irf, measured, t, channelwidth, tau=tau, amp=amp, shift=shift, ploton=True)




