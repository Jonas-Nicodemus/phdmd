# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'pHDMD'
copyright = '2022, Jonas Nicodemus'
author = 'Jonas Nicodemus'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinxcontrib.katex',
              'sphinxcontrib.bibtex',
]

import sphinxcontrib.katex as katex

latex_macros = r"""
    \def \i                {\mathrm{i}}
    \def \e              #1{\mathrm{e}^{#1}}
    \def \vec            #1{\mathbf{#1}}
    \def \x                {\vec{x}}
    \def \d                {\operatorname{d}\!}
    \def \dirac          #1{\operatorname{\delta}\left(#1\right)}
    \def \scalarprod   #1#2{\left\langle#1,#2\right\rangle}
    \def \timeStep          {\delta_t}
    \def \nrSnapshots       {M}
    \def \nrSnapshotsGen    {\tilde{\nrSnapshots}}
    \def \state             {x}
    \def \stateVar          {\state}
    \def \stateDim          {n}
    \def \stateDimRed       {r}
    \def \inpVar            {u}
    \def \inputVar          {\inpVar}
    \def \inpVarDim         {m}
    \def \inputDim          {\inpVarDim}
    \def \outVar            {y}
    \def \outputVar         {\outVar}
    \def \dmdV              {X_+}
    \def \dmdW              {X_{-}}
    \def \dmdY              {Y}
    \def \dmdU              {U}
    \def \dataZ             {\mathcal{Z}}
    \def \dataT             {\mathcal{T}}
    \def \dmdJJ             {\widetilde{\mathcal{J}}}
    \def \dmdRR             {\widetilde{\mathcal{R}}}
"""

# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = 'macros: {' + katex_macros + '}'

# Add LaTeX macros for LATEX builder
latex_elements = {'preamble': latex_macros}

bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# # Material theme options (see theme.conf for more information)
# html_theme_options = {
#
#     # Set the name of the project to appear in the navigation.
#     'nav_title': 'pHDMD',
#
#     # Set you GA account ID to enable tracking
#     'google_analytics_account': 'UA-XXXXX',
#
#     # Specify a base_url used to generate sitemap.xml. If not
#     # specified, then no sitemap will be built.
#     'base_url': 'https://project.github.io/project',
#
#     # Set the color and the accent color
#     'color_primary': 'blue',
#     'color_accent': 'light-blue',
#
#     # Set the repo location to get a badge with stats
#     # TODO: change when release
#     'repo_url': 'https://github.com/Jonas-Nicodemus/crispy-pancake',
#     'repo_name': 'crispy-pancake',
#
#     # Visible levels of the global TOC; -1 means unlimited
#     'globaltoc_depth': 3,
#     # If False, expand all TOC entries
#     'globaltoc_collapse': False,
#     # If True, show hidden TOC entries
#     'globaltoc_includehidden': False,
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']