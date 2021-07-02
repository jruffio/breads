from setuptools import setup, find_packages
from pkg_resources import parse_requirements
from os.path import dirname
from py.path import local

directory_name = dirname(local(__file__))
with open(directory_name + "/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(directory_name + '/requirements.txt') as requirements:
    setup(
        name='breads',
        version='0.0.1',
        author="Shubh Agrawal, Jean-Baptiste Ruffio",
        author_email="shubh@caltech.edu, jruffio@caltech.edu",
        description="Broad Respository for Exoplanet Analysis, Discovery, and Spectroscopy",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jruffio/breads",
        packages=find_packages(),
        install_requires=requirements.readlines(),
        include_package_data=True,
        package_data={'': ['data/*']},
    )
    