How to get good fits
====================

There is endless literature on the topic of TCSPC decay fitting; what follows is some tips as to what works with Full SMS
and the types of data it was designed to analyse.

*   Full SMS offes two types of fitting: non-linear least-squares and maximum likelihood. The former is highly robust
    and computationally efficient, however it delivers incorrect results at low photon numbers, where ML is
    preferred due to its correct accounting for Poisson noise (LS assumes Gaussian noise).
*   A detailed comparison of the performance of our implementations of ML an LS is still in the works, however,
    literature suggests decays with less than 20 000 photons should be fitted with ML [#]_.
*   In general,
    both these optimization techniques require good initial guess values for parameters, so it is a good idea to use
    the interactive tool to find parameters that roughly fit the decay before running the fit.
*   If you are using a measured IRF, make sure it contains enough counts so that there is no discernable noise
    in the decay. Noise in the IRF will translate to noise in the fitted decay.
*   Start with a 1-component fit and move on to 2 and from there to 3 components if the fit is not satisfactory.
    Overfitting typically results in components with idential lifetimes or tiny amplitudes.
*   If you have certain expected values for parameters, it might help to set bounds that reflect this. Parameters can
    also be fixed - one good use case for this is to analyse all your data and calculate the average IRF shift (and
    fwhm if you are simulating it) and then analyse the data again with those parameters fixed. Another good use is if
    there is a very short lifetime component (< IRF fwhm) - you can find the average lifetime of this component and then
    fix it before analysing the data again.
*   The startpoint and endpoint of the fit are important. Play around with the different options for the automatic
    startpoint to see what works. The endpoint is chosen as the channel with 20 times the background or 1% of the
    maximum, whichever is greater. This can be changed in the settings.
*   To evaluate goodness-of-fit, the best is visual inspection of the residuals (they should appear random and evenly
    distributed), as well as the Durbin-Watson parameter. The DW parameter is a measure of autocorrelation in the residuals and
    takes values between 0 and 4, with a value of 2 indicating no autocorrelation (in our case, this means a good fit).
    Values below 2 indicate possible positive autocorrelation [#]_, with a certain statistical significance. For example,
    a value of 1.9 might indicate autocorrelation with only 5% significance, while a value of 1.8 might indicate
    autocorrelation to 1% significance [#]_. The distribution of values changes with different numbers of datapoints and
    parameters, and therefore Full SMS calculates the significance values for each fit according to <link>. In our
    experience and consistent with literature, a good fit does not necessarily mean zero autocorrelation, but the DW
    bounds provide an objective and consistent metric for automatically filtering out bad fits. We have found that
    sometimes good fits have DW parameters lower than the 1% bound and therefore Full SMS provides 0.3% and 0.1% bounds
    as well. Note that the DW parameter is quite sensitive to deviations
    that are commonly found around the beginning and end of the decay. Therefore, it is usually necessary to set the
    startpoint and endpoint as described above, with a higher statistical significance for lower values.

.. [#] Maus, M. et al. Anal. Chem. 73, 2078–2086 (2001) https://doi.org/10.1021/ac000877g.
.. [#] Values above 2 indicate negative autocorrelation, which is not normally found in the residuals.
.. [#] More precisely, the values are lower bounds on the critical values of the distribution, since the exact values
       cannot generally be determined, for details see Turner, P. Applied Economics Letters 27, 1495–1499 (2020)
       https://doi.org/10.1080/13504851.2019.1691711.
.. |scipy.optimize.curve_fit| replace:: ``scipy.optimize.curve_fit``
.. _scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit