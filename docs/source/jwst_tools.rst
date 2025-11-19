.. _jwst_tools:

JWST tools
==========

BREADS includes several tools for working with JWST data. These include tools for observation planning, for running pipeline reductions of JWST data prior to BREADS processing, and for analysing and plotting the resulting outputs. 

This code is located within the `breads.jwst_tools` module and its submodules. This is all supporting code, *not* part of the "core" BREADS functionality.

Planning JWST observations: breads.jwst_tools.planning
-------------------------------------------------------

For planning JWST IFU observations, the functions `visualize_nrs_fov` and `visualize_miri_mrs_fov` display the FOV of an observation of a star and companion to help planning special requirements for position angles and offstes. See `this demo notebook <https://github.com/jruffio/breads/blob/main/demos/demo_planning_jwst_ifu_obs.ipynb>`_

Reducing JWST data: breads.jwst_tools.reduction_utils
-----------------------------------------------------

Functions in this module can be used to automate JWST pipeline reductions along with some custom steps and optimizations. See tutorial notebooks for JWST data analyses in this repo.


Plotting JWST data: breads.jwst_tools.plotting
----------------------------------------------

This module contains some (simple!) functions for making nice plots of JWST 2D images, mostly to help display the status of data files during reductions.

.. automodapi:: breads.jwst_tools
   :no-inheritance-diagram:
