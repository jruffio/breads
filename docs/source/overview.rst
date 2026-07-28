Quick Overview
==============

**BREADS** --- the **B**\road **R**\epository for **E**\xoplanet **A**\nalysis, **D**\etection, and **S**\pectroscopy ---
is an open-source Python package primarily aimed at processing moderate (R>1,000) to high-resolution (R>10,000) spectroscopic data of directly imaged exoplanets and brown dwarf companions.
It implements a framework to jointly model planet signal and starlight to detect and characterize faint companions hidden in the glare of their host stars (i.e., high-contrast), leveraging the spectral differences between the planet and the star.

Source code: `github.com/jruffio/breads <https://github.com/jruffio/breads>`_

Refer to the `installation instructions <installation.html>`_ for how to install BREADS.

What problem does BREADS solve?
--------------------------------

BREADS provides tools to compute:

- **Detection maps** Signal-to-noise (S/N) as a function of position.
- **Cross-correlation functions** S/N as a function of radial velocity, e.g. for molecular detection.
- **Likelihoods definition** for atmospheric characterization (e.g., T_eff, log g, C/O, etc.)
- **Planet spectral extraction** in specific cases.

The key idea is that the planet signal and the starlight can typically be modeled as linear combinations of basis functions (e.g., a spline for the stellar halo, principal components for residual speckles, and a forward model for the planet signal).
The typical issue would be that there can be 100s or even 1000s of linear parameters, which would make MCMC sampling intractable.
Assuming gaussian noise, the amplitudes of these linear components can be analytically estimated and marginalized, making the problem not only tractable but very fast.
This lets the user focus on the nonlinear parameters of interest (e.g., planet position, radial velocity, effective temperature, etc.), while the linear parameters are marginalized on the fly.
BREADS provides a flexible framework that can be adapted to various instruments and data types, enabling users to extract maximum information from their observations while accounting for complex noise and systematics.

Core framework philosophy
---------------

The user selects three building blocks:

1. **A data class** --- normalizes the data format across different instruments while encoding instrument-specific behaviors.
2. **A forward model (FM)** --- reproduces the data as ``d = FM + n``, where ``d`` is the observed data, ``FM`` is the forward model of the planet signal and starlight, and ``n`` is the noise. The FM accounts for the planet signal, the stellar halo, telluric contamination, and nuisance terms.
3. **A fitting strategy** --- e.g., grid search (for detection maps or cross-correlation functions), general optimizers (Nelder-Mead), or posterior samplers (MCMC).


.. note ::

   Data classes and forward models were originally designed to be mixed and matched, but it turns out forward models need to be specific to the instrument to address the specific quirks of each instrument.

Supported instruments
---------------------

BREADS includes tools for the following instruments. However, BREADS is constantly evolving and may not be stable.
At this time, we recommend that you reach out to the authors for more specific advice about your needs.

.. list-table::
   :header-rows: 1
   :widths: 25 30 20

   * - Instrument
     - Observatory
     - Resolution
   * - OSIRIS
     - Keck I
     - R = 4000
   * - KPIC
     - Keck II
     - R = 35,000
   * - NIRSpec IFU
     - JWST
     - R = 2700
   * - NIRSpec fixed-slit
     - JWST
     - R = 2700
   * - MIRI MRS
     - JWST
     - R = 2700

Citing BREADS
----------------

If you use BREADS in your research, please cite the following papers depending on the context:

In all cases, cite the ASCL record for the BREADS software itself:
`Ruffio et al. (2025) ASCL <https://ui.adsabs.harvard.edu/abs/2025ascl.soft01009R/abstract>`_

For **OSIRIS** data analysis, please cite:
`Agrawal et al. (2023), AJ, 166, 15 <https://ui.adsabs.harvard.edu/abs/2023AJ....166...15A/abstract>`_

For **KPIC** data analysis, please cite:
`Ruffio et al. (2023), AJ, 165, 113 <https://ui.adsabs.harvard.edu/abs/2023AJ....165..113R/abstract>`_

For **JWST NIRSpec IFU** data analysis, please cite:
`Ruffio et al. (2024), AJ, 168, 73 <https://ui.adsabs.harvard.edu/abs/2024AJ....168...73R/abstract>`_

For **JWST NIRSpec fixed-slit** data analysis, please cite:
`Madurowicz et al. (2025), AJ, 170, 326 <https://ui.adsabs.harvard.edu/abs/2025AJ....170..326M/abstract>`_

For **JWST MIRI** data analysis, please cite:
Bidot et al. (in prep.)

Other relevant references
----------------

Other relevant papers that use BREADS include (non-exhaustive list):

- `Ruffio et al. (2019), AJ, 158, 200 <https://ui.adsabs.harvard.edu/abs/2019AJ....158..200R/abstract>`_ --- foundational forward-model formalism and HR 8799 RVs with OSIRIS
- `Ruffio et al. (2021), AJ, 162, 290 <https://ui.adsabs.harvard.edu/abs/2021AJ....162..290R/abstract>`_ --- atmospheric characterization of HR 8799 b/c/d with OSIRIS
- `Sappey et al. (2023), AJ, 169, 175 <https://ui.adsabs.harvard.edu/abs/2025AJ....169..175S/abstract>`_ --- Atmospheric characterization of HD 206893 B with KPIC
- `Horstman et al. (2024), AJ, 168, 175 <https://ui.adsabs.harvard.edu/abs/2024AJ....168..175H/abstract>`_ --- Exomoon search around GQ Lup B with KPIC



