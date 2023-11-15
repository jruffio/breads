Overview and core concepts
==========================

The mathematical approach of ``breads`` is presented in `Agrawal et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023AJ....166...15A/abstract>`_. 
We reiterate here, adapted from section 3.2 of that paper: 

``breads``, or the Broad Repository for Exoplanet Analysis, Detection, and
Spectroscopy, is a flexible framework that allows forward modeling of data from
moderate to high-resolution spectrographs. The philosophy of ``breads`` is to have
the users choose a *data class*, a *forward model function*, and a *fitting
strategy*. 

:ref:`data_classes` normalize the data format, simplifying reduction across
different spectrographs while allowing for specific behaviors of each
instrument to also be coded into their own specific class. 

The :ref:`forward model
(FM) <forward_models>` aims to reproduce the data `d` as `d = FM + n`, where `n` is the noise. 
The FM is a function not only of relevant *astrophysical parameters* of the
planet and the host star but also some *nuisance parameters*. For a general FM
within `breads`, nuisance parameters do not contain physical information about
the planet but are needed to model the data accurately. For example, 
for the specific FM used in Agrawal et al. 2023,  the linear parameters that
model the spurious contribution of the host star, the contribution from
telluric-only component, and the contribution from the residual principal
components are all nuisance parameters. Meanwhile, planetary characteristics
(which are needed to model its spectrum) such as effective temperature, surface
gravity, and radial velocity or its position relative to the star are normal
astrophysical parameters and not nuisance parameters.

We distinguish between *linear and nonlinear parameters* in any forward model
function used within the ``breads`` framework because ``breads`` performs an
analytical marginalization of all of its linear parameters, as described in
Ruffio et al. (2019), to improve the tractability of the problem. For the
specific FM used in Agrawal et al. 2023, the contribution from each FM component 
is a linear parameter. Indeed, the posteriors for these linear
parameters can be calculated analytically without a sampling algorithm such as
Markov Chain Monte Carlo (MCMC), allowing for increased speed, and
higher-dimensional or complex models (Ruffio et al. 2019, 2021). The definition
of a data structure and a forward model leads to the definition of a likelihood
assuming Gaussian white noise, which can then be used to either optimize the
parameters through a maximum likelihood or derive their posteriors.

Examples of :ref:`fitting` include a simple grid search optimization, more
general optimizers (e.g., Nelder-Mead), or even posterior sampling algorithms
such as MCMC. The grid search can, for example, be used to compute detection
maps or cross-correlation functions by varying, respectively, the position of
the planet or its RV.


