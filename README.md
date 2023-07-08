# GalVel_Pop
Galactic Velocity Components and Stellar Populations



# Description

This code calculates the UVW Galactic Space Velocity components based on the method presented in [Johnson \& Soderblom (1987)](https://ui.adsabs.harvard.edu/abs/1987AJ.....93..864J/abstract).

The velocities can be computed with respect to the Sun or the Local Standard of Rest (LSR). In the latter case, the solar motion relative to the LSR is obtained from [Schönrich et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.403.1829S/abstract).

Next, the program assigns each star (with a given probability) to one of three populations: the thin disk, thick disk, or halo. This assignment follows the method described in [Reddy et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.367.1329R/abstract). It assumes that the stars in the solar neighborhood consist of a mixture of these three populations, and each population follows a Gaussian distribution of random velocities in each component (Schwarzschild 1907).

The mean values (asymmetric drift), velocity dispersion (characteristic velocity dispersion), and population fractions are taken from [Bensby et al (2005)](https://ui.adsabs.harvard.edu/abs/2005A%26A...433..185B/abstract) and [Robin et al. (2003)](https://ui.adsabs.harvard.edu/abs/2003A%26A...409..523R/abstract). It is also possible to use the mean values from both sources.

The uncertainties of the parameters are calculated by bootstrapping the input parameters within the range of their uncertainties.

This method has been utilized and documented in various publications, including [Adibekyan et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012A%26A...545A..32A/abstract).