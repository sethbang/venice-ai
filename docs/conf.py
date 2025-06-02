# conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Check for an environment variable to signify an AI-friendly build
AI_FRIENDLY_BUILD = os.getenv('AI_FRIENDLY_BUILD', 'false').lower() == 'true'

# -- Project information -----------------------------------------------------
project = 'venice-ai Python Client Library'
copyright = '2023-2025, The Venice AI Team'
author = 'The Venice AI Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # To parse Google and NumPy style docstrings
]

if not AI_FRIENDLY_BUILD:
    extensions.extend([
        'sphinx.ext.viewcode',    # To add links to source code
        'sphinx.ext.intersphinx', # Link to other projects' docs
    ])

# Configure Napoleon for Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
# html_static_path = ['_static']

# -- Options for intersphinx extension ---------------------------------------
if not AI_FRIENDLY_BUILD:
    intersphinx_mapping = {
        'python': ('https://docs.python.org/3', None),
        # 'httpx': ('https://www.encode.net/', None), # TODO: Find correct objects.inv URL for httpx
    }
else:
    intersphinx_mapping = {} # Or remove this block entirely if AI_FRIENDLY_BUILD is true