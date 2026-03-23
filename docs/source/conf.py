# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EnergyMiddleware'
copyright = '2026, Samuel van den Houten, Jan Kuhta, Andrea Vezzuto, Aadesh Ramai, Rodrigo Montero González'
author = 'Samuel van den Houten, Jan Kuhta, Andrea Vezzuto, Aadesh Ramai, Rodrigo Montero González'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))