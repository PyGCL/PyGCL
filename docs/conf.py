import GCL
import datetime
import sphinx_rtd_theme

project = 'PyGCL'
author = 'Yanqiao Zhu'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = GCL.__version__
release = GCL.__version__

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

templates_path = ['_templates']

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 2,
    'display_version': True,
}

html_static_path = ['_static']
