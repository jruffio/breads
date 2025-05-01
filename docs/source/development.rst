Development Guide
=================


Notes for how to develop breads can go here.


Building Documentation
----------------------

 - Make sure you have sphinx installed, and the other dependencies needed to build the docs. This can be done automatically via ``pip install 'breads[docs]'``. The extra ``[docs]`` tells pip to find and install the optical dependencies needed for the docs. (See `pyproject.toml <https://github.com/jruffio/breads/blob/main/pyproject.toml>`_ for details.)
 - In the ``breads/docs/source`` folder, edit the docs file as needed. See `here for docs on RST text syntax <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.
 - In ``breads/docs``, run ``make html`` . This should run and print a message like "The HTML pages are in build/html"
 - Verify those HTML pages from the edited docs look good. Iterate as needed.
 - Commit the changes to git in a new branch and make a pull request.
 - Once the pull request is reviewed and merged to main, the readthedocs will update automatically.
