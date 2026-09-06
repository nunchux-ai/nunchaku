# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path

project = "Nunchaku"
copyright = "2025, Nunchaku Team"
author = "Nunchaku Team"

version_path = Path(__file__).parent.parent.parent / "nunchaku" / "__version__.py"
version_ns = {}
exec(version_path.read_text(), {}, version_ns)
version = version_ns["__version__"]
version = version.replace("dev", "")
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "sphinx.ext.extlinks",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Include global link definitions -----------------------------------------
links_dir = Path(__file__).parent / "links"
rst_epilog = ""
if links_dir.exists() and links_dir.is_dir():
    for link_file in sorted(links_dir.glob("*.txt")):
        with open(link_file, encoding="utf-8") as f:
            rst_epilog += f.read()

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

extlinks = {
    "nunchaku_issue": ("https://github.com/nunchux-ai/nunchaku/issues/%s", "nunchaku#%s"),
    "comfyui-nunchaku_issue": ("https://github.com/nunchux-ai/ComfyUI-nunchaku/issues/%s", "ComfyUI-nunchaku#%s"),
}

html_favicon = "_static/nunchaku.ico"

html_theme_options = {
    "repository_url": "https://github.com/nunchux-ai/nunchaku",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    # "announcement": "🔥 Nunchaku v1.2 released!",
}

intersphinx_mapping = {
    "comfyui_nunchaku": ("https://nunchaku.tech/docs/ComfyUI-nunchaku", None),
}
