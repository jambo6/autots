import os
import sys

sys.path.insert(0, os.path.abspath("../"))


# Info
project = "autots"
copyright = "2021, James Morrill"
author = "James Morrill"
release = "v0.0.3"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
]

# Things
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# Dont include the full modules path
add_module_names = False

# Dont do alphabetical
autodoc_member_order = "bysource"

# Todos
todo_include_todos = True

# Use an edited version of the catalyst theme
# Answer taken from
# https://stackoverflow.com/questions/14622698/customize-sphinxdoc-theme
html_theme = "theme"  # use the theme in subdir 'theme'
html_theme_path = ["."]  # make sphinx search for themes in current dir

# Themes
# html_theme = "karma_sphinx_theme"
# html_theme = 'catalyst_sphinx_theme'
# html_theme = 'pytorch_sphinx_theme'
# html_theme = 'sphinx_rtd_theme'
