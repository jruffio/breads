Mathematical framework
==========================

The mathematical approach of ``BREADS`` was first presented in `Ruffio et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019AJ....158..200R/abstract>`_.
The python package ``BREADS`` was introduced in `Agrawal et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023AJ....166...15A/abstract>`_.

The framework has been constantly evolving since then. For example:

- Agrawal et al. 2023 also introduced the idea of splines to model the spectral continuum.
- `Ruffio et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021AJ....162..290R/abstract>`_ introduced the idea of including principal components of the residuals as part of the linear model to better fit for residual systematics (Inspired by approaches like molecular mapping `Hoeijmakers et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018A%26A...617A.144H/abstract>`_).
- `Ruffio et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023AJ....165..113R/abstract>`_ showed that the planet signal itself could also be decomposed as different linear components.
- `Ruffio et al. 2024 <https://ui.adsabs.harvard.edu/abs/2024AJ....168...73R/abstract>`_ introduced the idea of regularization (priors) on the linear parameters.

Linear Model and Marginalization
----------------------------------

The core of ``BREADS`` is linear least squares with Gaussian noise. Linear parameters can be fitted and marginalized analytically, which allows for fast inference of nonlinear parameters even when the number of linear parameters is large.
The data is modeled as (Ruffio et al. 2019, Appendix D):

.. math::

   \mathbf{d} = M_\psi \mathbf{\phi} + \mathbf{n}

where:

- :math:`\mathbf{d}` is the data vector of length :math:`N`,
- :math:`M_\psi` is the model matrix determined by the nonlinear parameters :math:`\psi`,
- :math:`\mathbf{\phi}` is the vector of :math:`N_\phi` linear parameters,
- :math:`\mathbf{n}` is a Gaussian random noise vector with zero mean and covariance matrix :math:`\Sigma = s^2 \Sigma_0`.

The corresponding Gaussian likelihood is:

.. math::

   \mathcal{L}(\psi, \mathbf{\phi}, s^2) = \frac{1}{\sqrt{(2\pi)^N |\Sigma|}}
   \exp\left\{-\frac{1}{2s^2}(\mathbf{d} - M\mathbf{\phi})^\top \Sigma_0^{-1} (\mathbf{d} - M\mathbf{\phi})\right\}

The best-fit linear parameters :math:`\tilde{\mathbf{\phi}}` are found analytically via the
pseudo-inverse (Ruffio et al. 2019, Eq. 16):

.. math::

   \tilde{\mathbf{\phi}} = (M^\top \Sigma_0^{-1} M)^{-1} M^\top \Sigma_0^{-1} \mathbf{d}

Their covariance matrix is (Ruffio et al. 2019, Eq. 23):

.. math::

   \text{cov}(\tilde{\mathbf{\phi}}) = (M^\top \Sigma^{-1} M)^{-1}

The uncertainties on the linear parameters (i.e., the square roots of the diagonal
elements of :math:`\text{cov}(\tilde{\mathbf{f}})`) are returned analytically by
:func:`~breads.fit.fitfm` without requiring any MCMC.
These uncertainties are also marginalized over all the other linear parameters in the model.



Python Implementation
----------------------------------

The philosophy of ``BREADS`` is to have the users define a *data class*, a *forward model function*, and a *fitting
strategy* (From section 3.2 of Agrawal et al. 2023).

:ref:`data_classes` defines the observation with attributes like the data, the noise standard deviation, the wavelength grid, and any other relevant information about the instrument or the observation.

The :ref:`forward model
(FM) <forward_models>` aims to reproduce the data :math:`d` as :math:`\mathbf{d} = M_\psi \mathbf{\phi} + \mathbf{n}`, where :math:`n` is the noise.
The FM (:math:`M_\psi`) is a function not only of relevant *astrophysical parameters* of the
planet and the host star but also some *nuisance parameters*. For a general FM
within `breads`, nuisance parameters do not contain physical information about
the planet but are needed to model the data accurately. For example,
for the specific FM used in Agrawal et al. 2023,  the linear parameters that
model the spurious contribution of the host star, and the contribution from the residual principal
components are all nuisance parameters. Meanwhile, planetary characteristics
(which are needed to model its spectrum and included in :math:`\psi`) such as effective temperature, surface
gravity, and radial velocity or its position relative to the star are normal
astrophysical parameters and not nuisance parameters.
The definition
of a data object and a forward model leads to the definition of a likelihood
assuming Gaussian white noise, which can then be used to either optimize the
parameters through a maximum likelihood or derive their posteriors.

Examples of :ref:`fitting` include a simple grid search optimization, more
general optimizers (e.g., Nelder-Mead), or even posterior sampling algorithms
such as MCMC. The grid search can, for example, be used to compute detection
maps or cross-correlation functions by varying, respectively, the position of
the planet or its RV.

Nonlinear Parameters and Grid Search
--------------------------------------

We distinguish between *linear* :math:`\mathbf{\phi}` and *nonlinear* parameters :math:`\psi` in any forward model
function used within the ``BREADS`` framework because ``BREADS`` performs an
analytical marginalization of all of its linear parameters.

Many physically meaningful parameters enter the forward model nonlinearly
(e.g., the radial velocity of a companion, the center of a spectral feature,
or atmospheric parameters such as effective temperature and surface gravity).

``BREADS`` handles nonlinear parameters :math:`\psi` by evaluating the
marginalized posterior on a discrete grid. For each grid point, :func:`~breads.fit.fitfm`
solves for the best-fit linear parameters analytically. The marginalized log-posterior
over the linear parameters :math:`\mathbf{\phi}` is computed at each grid point and returned as the log-probability
surface. For example, Eq. 41 in Ruffio et al. 2019 which also include the marginalization over the noise scaling factor:

The function :func:`~breads.grid_search.grid_search` systematically evaluates this
log-probability over a grid of nonlinear parameter values and returns the
log-probability map together with the best-fit linear parameters and their uncertainties
at each grid point. Confidence intervals for the nonlinear parameters are then derived
from the resulting posterior distribution using :func:`~breads.utils.get_err_from_posterior`.

The BREADS marginalized posterior for the nonlinear parameters is fully consistent
with a traditional MCMC approach, but is more
computationally efficient when linear parameters are numerous.
It is possible to use the BREADS marginalized
log-probability directly as the log-likelihood inside an MCMC sampler via
:func:`~breads.fit.log_prob`, which allows sampling the nonlinear parameters
while analytically marginalizing over all linear parameters on the fly.


Noise Scaling Factor
---------------------

In practice, the noise vector :math:`\mathbf{s}` supplied in the data object may be
inaccurate --- for example, the noise may be systematically underestimated due to
correlated residuals in high-contrast imaging.
``BREADS`` provides two complementary approaches to handle this.

**Empirical noise rescaling** (``scale_noise=True`` in :func:`~breads.fit.fitfm`): The
noise vector is rescaled by :math:`\sqrt{\tilde{\chi}^2_\text{red}}`, the square root
of the reduced chi-squared of the best-fit model. This corrects the uncertainties on
the linear parameters so they are consistent with the amplitude of the residuals.
Note that this does not propagate the uncertainty of the noise scaling factor into the
log-probability used for nonlinear parameter inference; but it can typically be neglected.

**Marginalization over the noise scaling factor** (``marginalize_noise_scaling=True``):
A more statistically rigorous approach to analytically marginalize the returned log probability of the fit over the noise scaling factor
(Ruffio et al. 2019, Eq. 40).

In practice, the noise scaling factor is typically well constrained by the data, and the
difference between these two approaches is often negligible. Note that
marginalization over the noise scaling factor is not compatible with regularization
(see below).

Regularization (Priors on Linear Parameters)
----------------------------------------------

``BREADS`` supports Gaussian priors (regularization) on the linear parameters
:math:`\mathbf{\phi}`, as described in Ruffio et al. (2024, Appendix A). Regularization
is useful when the data are not sufficient to fully constrain all linear parameters,
or when prior knowledge (e.g., from a reference differential imaging observation)
is available to constrain the background or continuum level.

A Gaussian prior on :math:`\mathbf{\phi}` with mean :math:`\boldsymbol{\mu}_\phi` and diagonal
covariance :math:`\Sigma_\text{reg}` is incorporated by augmenting the data vector,
the model matrix, and the noise vector (Ruffio et al. 2024, Eq. A4):

.. math::

   \mathbf{d}' = \begin{bmatrix} \mathbf{d} \\ \boldsymbol{\mu}_\phi \end{bmatrix}, \quad
   M' = \begin{bmatrix} M \\ M_\text{reg} \end{bmatrix}, \quad
   \Sigma' = \begin{bmatrix} \Sigma & 0 \\ 0 & \Sigma_\text{reg} \end{bmatrix}

where :math:`M_\text{reg}` is a selection matrix (identity or partial identity) that
selects the subset of parameters being regularized.

The maximum a posteriori estimate is then given by the same pseudo-inverse expression
as without regularization, but applied to the augmented system (Ruffio et al. 2024, Eq. A5):

.. math::

   \tilde{\mathbf{\phi}} = (M'^\top \Sigma'^{-1} M')^{-1} M'^\top \Sigma'^{-1} \mathbf{d}'

The covariance of :math:`\tilde{\mathbf{\phi}}` is updated accordingly (Ruffio et al. 2024, Eq. A10):

.. math::

   \text{cov}(\tilde{\mathbf{\phi}}) = (M'^\top \Sigma'^{-1} M')^{-1}
   (M^\top \Sigma^{-1} M)
   (M'^\top \Sigma'^{-1} M')^{-1}
