import GCL
import datetime
import sphinx_rtd_theme

project = 'PyGCL'
author = 'Yanqiao Zhu'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = GCL.__version__
release = GCL.__version__

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
]
autosummary_generate = True

myst_enable_extensions = [
    'smartquotes',
    'colon_fence',
    'dollarmath',
    'deflist',
    'substitution',
]
myst_dmath_double_inline = True
myst_heading_anchors = 2

intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

templates_path = ['_templates']

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 2,
    'display_version': True,
}
add_module_names = False

rst_context = {'GCL': GCL}


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect('autodoc-skip-member', skip)
    app.connect('source-read', rst_jinja_render)
