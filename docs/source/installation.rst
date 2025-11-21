Installation
============

breads installs in the usual manner for Python packages. 

.. note ::

   More detailed installation instructions may eventually go here. But really, it's pretty straightforward.

From github
------------
Briefly: 

 * ``git clone git@github.com:jruffio/breads.git``
 * ``cd breads``
 * ``pip install .``

**Installing with optional dependencies**

BREADS may be used to reduce data from several facilities, and some dependencies are optional depending on what kind of data you will be reducing. For instance, the JWST pipeline (python package ``jwst``) is only a required dependency if you're going to be reduing JWST data.  Optional dependencies can be specified using brackets in the ``pip`` command line.

 * For using BREADS with JWST data: ``pip install .[jwst]``

 * For installing tools to build the HTML package documentation: ``pip install .[docs]``

From pypi
----------

Not currently recommended; the version of BREADS on pypi is out of date compated to the current github development version.
