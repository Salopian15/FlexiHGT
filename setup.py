"""Compatibility shim.

All packaging metadata now lives in pyproject.toml; keeping it in two places is
how the version, URL and dependency list drifted apart. Install as usual with
`pip install .`.
"""

from setuptools import setup

setup()
