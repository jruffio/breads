.. breads documentation master file, created by
   sphinx-quickstart on Sun Dec 19 10:04:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

breads, the Broad Repository for Exoplanet Analysis, Discovery, and Spectroscopy
================================================================================


``breads`` is a toolkit for data analyses in astronomical spectroscopy of
exoplanets, in particular frameworks for rigorous forward modeling of
observational data to achieve physical inferences with reduced systematic biases.
``breads`` currently has specific functionality for modeling data from JWST NIRSpec, Keck OSIRIS, and Keck KPIC, but the
underlying mathematical framework is more general.

Breads has been developed by Jean-Baptiste Ruffio (UC San Diego) and collaborators.


.. warning::

   This is very incomplete partial documentation for a work-in-progress, pre-1.0 research software project.
    

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   overview.rst
   installation.rst
   data_classes.rst
   forward_models.rst
   fitting.rst
   utils.rst
   development.rst


In addition to these docs, there are also separate repositories with example code and notebooks for particular datasets:

 * `JWST NIRSpec on HD 19467 <https://github.com/jruffio/HD_19467_B>`_, showing the analyses from Ruffio et al. 2024
 * `JWST NIRSpec point spread function reference stars <https://github.com/jruffio/NIRSpec_PSF_calib_3399>`_, from JWST calibration program 3399



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
