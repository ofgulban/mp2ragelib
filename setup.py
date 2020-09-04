"""mp2ragelib setup.

To install, using the commandline do:
    pip install -e /path/to/mp2ragelib

Notes for PyPI:
---------------
python setup.py sdist upload -r pypitest
python setup.py sdist upload -r pypi

"""


from setuptools import setup

VERSION = '0.0.0'

setup(name="mp2ragelib",
      version=VERSION,
      description="MP2RAGE stuff from Marques et al. (2010).",
      url='https://github.com/ofgulban/mp2ragelib',
      download_url=('https://github.com/ofgulban/mp2ragelib/archive/'
                    + VERSION + '.tar.gz'),
      license="BSD-3-clause",
      author="Omer Faruk Gulban",
      packages=['mp2ragelib'],
      install_requires=['numpy', 'nibabel'],
      # entry_points={'console_scripts': [
      #     'pymp2rage = mp2ragelib.pymp2rage:main']},
      zip_safe=True)
