"""For having the version."""

import pkg_resources

__version__ = pkg_resources.require("mp2ragelib")[0].version
