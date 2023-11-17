# minimal setup.py
#
from setuptools import setup

setup(name='talar',
      version='0.0.1',
      install_requires=[],
      packages=['talar'],
      package_dir={'talar': 'talar'},
      package_data={'talar': ['grammar/*.txt']},
      include_package_data=True,
)
