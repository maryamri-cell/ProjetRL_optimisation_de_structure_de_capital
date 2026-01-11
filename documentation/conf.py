# Configuration file for Sphinx documentation builder.

project = 'Capital Structure RL Optimizer'
copyright = '2025, RL Finance Team'
author = 'RL Finance Team'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx_markdown_tables',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

import os

# ReadTheDocs theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
}

html_static_path = ['_static']
if os.path.exists(os.path.join(os.path.dirname(__file__), '_static', 'logo.png')):
    html_logo = '_static/logo.png'
else:
    html_logo = None
html_title = "Capital Structure RL Optimizer"

# Add custom CSS
html_css_files = [
    'css/custom.css',
]

# MyST parser config
myst_enable_extensions = ["dollarmath", "amsmath"]
source_suffix = {
    '.rst': None,
    '.md': 'myst',
}
