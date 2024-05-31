import os
import sys

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)


project = "OnnxSlim"
copyright = "2022, WeLoveAI"
author = "inisis"

import onnxslim

version = onnxslim.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
epub_show_urls = "footnote"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
napoleon_preprocess_types = True

html_static_path = ["_static"]


def setup(app):
    """Configure the Sphinx app by adding a custom CSS file ('style.css')."""
    app.add_css_file("style.css")
