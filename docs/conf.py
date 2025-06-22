# conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Check for an environment variable to signify an AI-friendly build
AI_FRIENDLY_BUILD = os.getenv('AI_FRIENDLY_BUILD', 'false').lower() == 'true'

# -- Project information -----------------------------------------------------
project = 'venice-ai Python Client Library'
copyright = '2025, venice-ai'
author = 'Seth Bang'
release = '1.2.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # To parse Google and NumPy style docstrings
    'sphinx_autodoc_typehints', # Add this line
]

if not AI_FRIENDLY_BUILD:
    extensions.extend([
        'sphinx.ext.viewcode',    # To add links to source code
        'sphinx.ext.intersphinx', # Link to other projects' docs
    ])

# Configure Napoleon for Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for sphinx-autodoc-typehints ------------------------------------
# This will automatically document type hints in the signature (recommended)
typehints_formatter = None # Uses the default formatter
# If True, type hints will be shown in the signature.
# If False, type hints will be shown in the description (less common for signatures).
typehints_use_signature = True
# If True, class names in type hints will be fully qualified (e.g. module.Class).
# If False, only the class name will be used (e.g. Class).
typehints_fully_qualified = True
# If True, will add | None to optional types.
always_document_param_types = True # Ensures param types are documented even if already in signature
# This option controls how to represent Union types. 'smart' is a good default.
# It simplifies Union[Optional[X], X] to X and Union[X, None] to Optional[X].
simplify_optional_unions = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
# html_static_path = ['_static']

# -- Options for intersphinx extension ---------------------------------------
if not AI_FRIENDLY_BUILD:
    intersphinx_mapping = {
        'python': ('https://docs.python.org/3', None),
    }
else:
    intersphinx_mapping = {} # Or remove this block entirely if AI_FRIENDLY_BUILD is true