from setuptools import setup, find_packages
from pkg_resources import parse_requirements

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('./requirements.txt') as requirements:
    setup(
        name='breads',
        version='0.0.1',
        author="Jean-Baptiste Ruffio, Shubh Agrawal",
        author_email="shubh@caltech.edu",
        description="Broad Respository for Exoplanet Analysis and DiscoverieS",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jruffio/breads",
        packages=find_packages(),
        install_requires=requirements.readlines()
    )