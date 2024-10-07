# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Adiabatic Taper'
copyright = '2024, Enrico Vallar'
author = 'Enrico Vallar'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    
]

# Configure Napoleon to support Google and NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# This prevents the __init__ method's docstring from being included in the class-level documentation
napoleon_include_init_with_doc = False

# Hide the full method signatures, show only the method name with (...)
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'no-signature': True,
    
}
autodoc_preserve_defaults = False



# Allow .md files to be treated as source files
source_suffix = ['.rst', '.md']

# Use class-level docstrings only, avoid constructor docstring unless specified explicitly
autoclass_content = 'class'

# Define the path to your source files
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

