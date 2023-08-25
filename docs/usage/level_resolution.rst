Intensity level resolution
==========================

Intensity levels can be resolved using change-point analysis in the "Intensity" tab. First choose a confidence level,
then click "Resolve" to resolve the current particle levels, "Resolve Selected" ot resolve all selected particles, or
"Resolve All" to resolve the levels for all particles. The change-point analysis uses the algorithm from Watkins and
Yang [#]_  and the confidence is a measure of how strict the algorithm is with deciding on a change point. A lower
confidence will therefore result in more fitted levels - it is up to the user to decide the best value for their
application.

.. [#] Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)