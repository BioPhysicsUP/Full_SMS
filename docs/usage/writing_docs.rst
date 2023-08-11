Documentation guide
===================

The documentation is built using `Sphinx <https://www.sphinx-doc.org/>`_, which uses `reStructuredText
<https://docutils.sourceforge.io/rst.html>`_, a simple but powerful markup language that was in large part created to
document Python code. A quick overview of reST can be found `here
<https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_.

Pages such as this one are written in pure reST while the documentation for modules and functions is built from
docstrings using `Sphinx AutoApi <https://sphinx-autoapi.readthedocs.io/en/latest/index.html>`_. We use the
`NumPy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

The source files for the documentation are in the main GitHub repository (in the ``docs`` folder) and the HTML
documentation is hosted on `Read the Docs <https://readthedocs.org/>`_. A GitHub hook automatically updates the
documentation when ``master`` is updated.

To build the documentation locally, first install the ``sphinx``, ``sphinx-rtd-theme`` and ``sphinx-autoapi`` packages
into your Python environment. Then ``cd`` into ``docs`` and execute ``make html``. Please do not add the resulting HTML
pages (in ``docs/_build``) to git.